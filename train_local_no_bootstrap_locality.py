from typing import Any
import jax.numpy as jnp
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import flax
import optax
import wandb
from ml_collections import config_flags
import ml_collections

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainStateEma
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from utils.sharding import create_sharding, all_gather
from utils.datasets import get_dataset
from model import DiT
from helper_eval import eval_model
from helper_inference import do_inference

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'imagenet256', 'Environment name.')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('fid_stats', None, 'FID stats file.')
flags.DEFINE_string('cluster_dir', None, 'Directory containing cluster assignments from encode_and_cluster.py')
flags.DEFINE_integer('seed', 10, 'Random seed.') # Must be the same across all processes.
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 32, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1_000_000), 'Number of training steps.')
flags.DEFINE_integer('debug_overfit', 0, 'Debug overfitting.')
flags.DEFINE_string('mode', 'train', 'train or inference.')

model_config = ml_collections.ConfigDict({
    'lr': 0.0001,
    'beta1': 0.9,
    'beta2': 0.999,
    'weight_decay': 0.1,
    'use_cosine': 0,
    'warmup': 0,
    'dropout': 0.0,
    'hidden_size': 64, # change this!
    'patch_size': 8, # change this!
    'depth': 2, # change this!
    'num_heads': 2, # change this!
    'mlp_ratio': 1, # change this!
    'class_dropout_prob': 0.1,
    'num_classes': 1000,
    'denoise_timesteps': 128,
    'cfg_scale': 4.0,
    'target_update_rate': 0.999,
    'use_ema': 1,
    'use_stable_vae': 1,
    'sharding': 'dp', # dp or fsdp.
    't_sampling': 'discrete-dt',
    'dt_sampling': 'uniform',
    'bootstrap_cfg': 0,
    'bootstrap_every': 8, # Make sure its a divisor of batch size.
    'bootstrap_ema': 1,
    'bootstrap_dt_bias': 0,
    'train_type': 'shortcut_local', # Use locality-aware loss
    # Locality-specific parameters
    'locality_noise_scale': 0.1,  # Scale of noise perturbation for locality constraint
    'locality_weight': 1.0,  # Weight for locality consistency loss
    'use_cluster_neighborhoods': False,  # Use cluster-based pairing instead of random perturbation
    'num_clusters': 500,  # Number of clusters (if using cluster neighborhoods)
})


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'shortcut',
    'name': 'local_{dataset_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('model', model_config, lock_config=False)
    
##############################################
## Training Code.
##############################################
def main(_):

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    print(f"Process index: {jax.process_index()}, Mode: {FLAGS.mode}")
    if jax.process_index() == 0 and FLAGS.mode == 'train':
        print("Initializing wandb...")
        try:
            setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)
            print(f"wandb initialized successfully. Run: {wandb.run}")
            print(f"wandb project: {FLAGS.wandb.project}, name: {FLAGS.wandb.name}")
        except Exception as e:
            print(f"wandb initialization failed: {e}")
            import traceback
            traceback.print_exc()
        
    dataset = get_dataset(FLAGS.dataset_name, local_batch_size, True, FLAGS.debug_overfit)
    dataset_valid = get_dataset(FLAGS.dataset_name, local_batch_size, False, FLAGS.debug_overfit)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]
    example_obs_shape = example_obs.shape
    
    # Load cluster assignments if using cluster neighborhoods
    cluster_assignment_map = None
    if FLAGS.model.use_cluster_neighborhoods and FLAGS.cluster_dir is not None:
        import csv
        from pathlib import Path
        assignments_path = Path(FLAGS.cluster_dir) / "assignments.csv"
        if assignments_path.exists():
            # Load assignments into a dictionary: image_index -> cluster_id
            cluster_assignment_map = {}
            with open(assignments_path, 'r') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    # Map by index since we don't have paths in iterator-based datasets
                    cluster_assignment_map[idx] = int(row['cluster'])
            print(f"Loaded {len(cluster_assignment_map)} cluster assignments from {assignments_path}")
            FLAGS.model.num_clusters = max(cluster_assignment_map.values()) + 1
            print(f"Using cluster neighborhoods with {FLAGS.model.num_clusters} clusters")
        else:
            print(f"Warning: assignments.csv not found at {assignments_path}, using random perturbation")
            FLAGS.model.use_cluster_neighborhoods = False
    
    if FLAGS.model.use_cluster_neighborhoods:
        print(f"Locality mode: Using cluster-based neighborhoods with {FLAGS.model.num_clusters} clusters")
    else:
        print(f"Locality mode: Using random noise perturbation (scale={FLAGS.model.locality_noise_scale})")

    if FLAGS.model.use_stable_vae:
        vae = StableVAE.create()
        if 'latent' in FLAGS.dataset_name:
            example_obs = example_obs[:, :, :, example_obs.shape[-1] // 2:]
            example_obs_shape = example_obs.shape
        else:
            example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        example_obs_shape = example_obs.shape
        vae_rng = jax.random.PRNGKey(42)
        vae_encode = jax.jit(vae.encode)
        vae_decode = jax.jit(vae.decode)

    if FLAGS.fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats
        get_fid_activations = get_fid_network() 
        truth_fid_stats = np.load(FLAGS.fid_stats)
    else:
        get_fid_activations = None
        truth_fid_stats = None

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs_shape[-1]
    FLAGS.model.image_size = example_obs_shape[1]
    dit_args = {
        'patch_size': FLAGS.model['patch_size'],
        'hidden_size': FLAGS.model['hidden_size'],
        'depth': FLAGS.model['depth'],
        'num_heads': FLAGS.model['num_heads'],
        'mlp_ratio': FLAGS.model['mlp_ratio'],
        'out_channels': example_obs_shape[-1],
        'class_dropout_prob': FLAGS.model['class_dropout_prob'],
        'num_classes': FLAGS.model['num_classes'],
        'dropout': FLAGS.model['dropout'],
        'ignore_dt': False,  # Locality method uses dt like shortcut
    }
    model_def = DiT(**dit_args)
    tabulate_fn = flax.linen.tabulate(model_def, jax.random.PRNGKey(0))
    print(tabulate_fn(example_obs, jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,), dtype=jnp.int32)))

    if FLAGS.model.use_cosine:
        lr_schedule = optax.warmup_cosine_decay_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'], FLAGS.max_steps)
    elif FLAGS.model.warmup > 0:
        lr_schedule = optax.linear_schedule(0.0, FLAGS.model['lr'], FLAGS.model['warmup'])
    else:
        lr_schedule = lambda x: FLAGS.model['lr']
    adam = optax.adamw(learning_rate=lr_schedule, b1=FLAGS.model['beta1'], b2=FLAGS.model['beta2'], weight_decay=FLAGS.model['weight_decay'])
    tx = optax.chain(adam)
    
    def init(rng):
        param_key, dropout_key, dropout2_key = jax.random.split(rng, 3)
        example_t = jnp.zeros((1,))
        example_dt = jnp.zeros((1,))
        example_label = jnp.zeros((1,), dtype=jnp.int32)
        example_obs = jnp.zeros(example_obs_shape)
        model_rngs = {'params': param_key, 'label_dropout': dropout_key, 'dropout': dropout2_key}
        params = model_def.init(model_rngs, example_obs, example_t, example_dt, example_label)['params']
        opt_state = tx.init(params)
        return TrainStateEma.create(model_def, params, rng=rng, tx=tx, opt_state=opt_state)
    
    rng = jax.random.PRNGKey(FLAGS.seed)
    train_state_shape = jax.eval_shape(init, rng)

    data_sharding, train_state_sharding, no_shard, shard_data, global_to_local = create_sharding(FLAGS.model.sharding, train_state_shape)
    train_state = jax.jit(init, out_shardings=train_state_sharding)(rng)
    jax.debug.visualize_array_sharding(train_state.params['FinalLayer_0']['Dense_0']['kernel'])
    jax.debug.visualize_array_sharding(train_state.params['TimestepEmbedder_1']['Dense_0']['kernel'])
    jax.experimental.multihost_utils.assert_equal(train_state.params['TimestepEmbedder_1']['Dense_0']['kernel'])
    start_step = 1

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        replace_dict = cp.load_as_dict()['train_state']
        del replace_dict['opt_state'] # Debug
        
        # Check if parameters have an extra leading dimension and strip it if needed
        def strip_leading_dim_if_needed(loaded_param, expected_param):
            # If loaded param has one more dimension at the front with size 1, strip it
            if len(loaded_param.shape) == len(expected_param.shape) + 1 and loaded_param.shape[0] == 1:
                return loaded_param[0]
            return loaded_param
        
        # Strip extra dimensions from params and params_ema if they exist
        if 'params' in replace_dict:
            replace_dict['params'] = jax.tree_util.tree_map(
                strip_leading_dim_if_needed, 
                replace_dict['params'],
                train_state.params
            )
        if 'params_ema' in replace_dict:
            replace_dict['params_ema'] = jax.tree_util.tree_map(
                strip_leading_dim_if_needed,
                replace_dict['params_ema'],
                train_state.params_ema
            )
        
        train_state = train_state.replace(**replace_dict)
        start_step = int(train_state.step.item())
        # Recompute sharding based on the loaded checkpoint's actual shapes
        loaded_train_state_shape = jax.eval_shape(lambda: train_state)
        _, train_state_sharding_loaded, _, _, _ = create_sharding(FLAGS.model.sharding, loaded_train_state_shape)
        # Reshard to convert numpy arrays back to JAX arrays with proper sharding
        try:
            train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding_loaded)(train_state)
            print("Loaded model with step", train_state.step)
            # Update the main sharding to match what was loaded
            train_state_sharding = train_state_sharding_loaded
        except Exception as e:
            print(f"Error resharding loaded checkpoint: {e}")
            print("This usually means the model configuration doesn't match the checkpoint.")
            print(f"Current config: hidden_size={FLAGS.model.hidden_size}, patch_size={FLAGS.model.patch_size}, depth={FLAGS.model.depth}")
            print("Please ensure the model configuration matches the checkpoint you're loading.")
            raise
        del cp

    train_state_teacher = None

    visualize_labels = example_labels
    visualize_labels = shard_data(visualize_labels)
    visualize_labels = jax.experimental.multihost_utils.process_allgather(visualize_labels)
    imagenet_labels = open('data/imagenet_labels.txt').read().splitlines()
    
    # Create a stateful batch counter for cluster assignment mapping
    batch_counter = {'count': 0}

    ###################################
    # Update Function
    ###################################

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update(train_state, train_state_teacher, images, labels, cluster_assignments=None, force_t=-1, force_dt=-1):
        new_rng, targets_key, dropout_key, perm_key = jax.random.split(train_state.rng, 4)
        info = {}

        id_perm = jax.random.permutation(perm_key, images.shape[0])
        images = images[id_perm]
        labels = labels[id_perm]
        # Also permute cluster assignments if provided
        if cluster_assignments is not None:
            cluster_assignments = cluster_assignments[id_perm]
        images = jax.lax.with_sharding_constraint(images, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        if FLAGS.model['cfg_scale'] == 0: # For unconditional generation.
            labels = jnp.ones(labels.shape[0], dtype=jnp.int32) * FLAGS.model['num_classes']

        # Import and use locality-aware targets
        from targets_shortcut_LFM_local import get_targets
        x_t, v_t, t, dt_base, labels_dropped, info = get_targets(
            FLAGS, targets_key, train_state, images, labels, force_t, force_dt, 
            cluster_assignments=cluster_assignments
        )

        def loss_fn(grad_params):
            v_prime, logvars, activations = train_state.call_model(x_t, t, dt_base, labels_dropped, train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True)
            mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
            
            # Store base mse_v before adding locality term
            mse_v_base = mse_v
            locality_loss = 0.0

            # Add locality consistency loss per-sample ONLY for flow-matching samples (not bootstrap)
            if 'locality_u_t' in info and 'locality_s_t' in info:
                u_t = info['locality_u_t']
                s_t = info['locality_s_t']
                locality_weight = info['locality_weight']
                locality_valid_mask = info.get('locality_valid_mask', None)
                
                bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
                bst_size_data = FLAGS.batch_size - bootstrap_size
                
                if bst_size_data > 0:
                    # Only use locality data for flow-matching portion (skip bootstrap)
                    # The locality data corresponds to the original full batch, so we need:
                    # - Skip first bootstrap_size entries in the locality data
                    # - Use only the flow-matching portion (bst_size_data entries)
                    u_t_flow = u_t[bootstrap_size:bootstrap_size+bst_size_data]
                    s_t_flow = s_t[bootstrap_size:bootstrap_size+bst_size_data]
                    
                    if locality_valid_mask is not None:
                        valid_mask_flow = locality_valid_mask[bootstrap_size:bootstrap_size+bst_size_data]
                    else:
                        valid_mask_flow = jnp.ones(bst_size_data, dtype=bool)
                    
                    # Predict velocities at locality points for flow-matching samples only
                    # Need to match the batch structure: we want to predict for the flow portion
                    # In the merged batch, flow samples are at indices [bootstrap_size:]
                    t_flow = t[bootstrap_size:]
                    dt_base_flow = dt_base[bootstrap_size:]
                    labels_flow = labels_dropped[bootstrap_size:]
                    
                    v_u_pred, _, _ = train_state.call_model(u_t_flow, t_flow, dt_base_flow, labels_flow, train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True)
                    v_s_pred, _, _ = train_state.call_model(s_t_flow, t_flow, dt_base_flow, labels_flow, train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True)
                    
                    # Compute per-sample locality MSE
                    locality_mse = jnp.mean((v_u_pred - v_s_pred) ** 2, axis=(1, 2, 3))
                    locality_mse_masked = jnp.where(valid_mask_flow, locality_mse, 0.0)
                    locality_loss = jnp.mean(locality_mse_masked)
                    
                    # Add locality MSE per-sample ONLY to flow-matching portion of mse_v
                    # Create a zero array for bootstrap portion and locality mse for flow portion
                    locality_contribution = jnp.concatenate([
                        jnp.zeros(bootstrap_size),  # No locality loss for bootstrap
                        locality_weight * locality_mse_masked  # Locality loss for flow-matching
                    ], axis=0)
                    mse_v = mse_v + locality_contribution

            # Now compute loss and split (like train.py)
            loss = jnp.mean(mse_v)
            bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
            
            loss_info = {
                'loss': loss,
                'loss_flow': jnp.mean(mse_v[bootstrap_size:]),
                'loss_bootstrap': jnp.mean(mse_v[:bootstrap_size]),
                'mse_v_base': jnp.mean(mse_v_base),
                'locality_loss': locality_loss,
                'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                **{'activations/' + k : jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()},
            }
            
            return loss, loss_info
        
        grads, new_info = jax.grad(loss_fn, has_aux=True)(train_state.params)
        info = {**info, **new_info}
        updates, new_opt_state = train_state.tx.update(grads, train_state.opt_state, train_state.params)
        new_params = optax.apply_updates(train_state.params, updates)

        info['grad_norm'] = optax.global_norm(grads)
        info['update_norm'] = optax.global_norm(updates)
        info['param_norm'] = optax.global_norm(new_params)
        info['lr'] = lr_schedule(train_state.step)

        train_state = train_state.replace(rng=new_rng, step=train_state.step + 1, params=new_params, opt_state=new_opt_state)
        train_state = train_state.update_ema(FLAGS.model['target_update_rate'])
        return train_state, info
    
    if FLAGS.mode != 'train':
        do_inference(FLAGS, train_state, None, dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                       get_fid_activations, imagenet_labels, visualize_labels, 
                       fid_from_stats, truth_fid_stats)
        return

    ###################################
    # Train Loop
    ###################################

    for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step),
                       smoothing=0.1,
                       dynamic_ncols=True):
        
        # Sample data.
        if not FLAGS.debug_overfit or i == 1:
            batch_images, batch_labels = shard_data(*next(dataset))
            if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                vae_rng, vae_key = jax.random.split(vae_rng)
                batch_images = vae_encode(vae_key, batch_images)
            
            # Get cluster assignments for this batch if using cluster neighborhoods
            batch_cluster_assignments = None
            if cluster_assignment_map is not None and FLAGS.model.use_cluster_neighborhoods:
                # Map batch indices to cluster IDs
                # Note: This assumes sequential iteration. For more complex datasets,
                # you may need to store image IDs with the dataset
                batch_size = batch_images.shape[0]
                start_idx = batch_counter['count']
                batch_cluster_ids = []
                for j in range(batch_size):
                    idx = (start_idx + j) % len(cluster_assignment_map)
                    batch_cluster_ids.append(cluster_assignment_map.get(idx, 0))
                batch_cluster_assignments = shard_data(jnp.array(batch_cluster_ids, dtype=jnp.int32))
                batch_counter['count'] += batch_size

        # Train update.
        train_state, update_info = update(
            train_state, train_state_teacher, batch_images, batch_labels, 
            cluster_assignments=batch_cluster_assignments
        )

        if i % FLAGS.log_interval == 0 or i == 1:
            update_info = jax.device_get(update_info)
            update_info = jax.tree_util.tree_map(lambda x: np.array(x), update_info)
            update_info = jax.tree_util.tree_map(lambda x: x.mean(), update_info)
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            valid_images, valid_labels = shard_data(*next(dataset_valid))
            if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                valid_images = vae_encode(vae_rng, valid_images)
            # For validation, we can optionally skip cluster assignments or use random assignment
            _, valid_update_info = update(
                train_state, train_state_teacher, valid_images, valid_labels, 
                cluster_assignments=None  # Skip clustering for validation
            )
            valid_update_info = jax.device_get(valid_update_info)
            valid_update_info = jax.tree_util.tree_map(lambda x: x.mean(), valid_update_info)
            train_metrics['training/loss_valid'] = valid_update_info['loss']

            if jax.process_index() == 0:
                print(f"Step {i}: Logging {len(train_metrics)} metrics to wandb")
                print(f"Sample metrics: loss={train_metrics.get('training/loss', 'N/A'):.4f}")
                if wandb.run is not None:
                    wandb.log(train_metrics, step=int(train_state.step.item()))
                    print(f"Successfully logged to wandb")
                else:
                    print("WARNING: wandb.run is None, not logging!")

        if i % FLAGS.eval_interval == 0:
            eval_model(FLAGS, train_state, train_state_teacher, int(train_state.step.item()), dataset, dataset_valid, shard_data, vae_encode, vae_decode, update,
                       get_fid_activations, imagenet_labels, visualize_labels, 
                       fid_from_stats, truth_fid_stats)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            train_state_gather = jax.experimental.multihost_utils.process_allgather(train_state)
            if jax.process_index() == 0:
                cp = Checkpoint(FLAGS.save_dir+str(train_state_gather.step+1), parallel=False)
                cp.train_state = train_state_gather
                cp.save()
                del cp
            del train_state_gather

if __name__ == '__main__':
    app.run(main)
