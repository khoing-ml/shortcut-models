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
from pathlib import Path

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
flags.DEFINE_string('cluster_dir', None, 'Directory containing cluster_centers.npy from encode_and_cluster.py')
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
    'use_ema': 0,
    'use_stable_vae': 1,
    'sharding': 'dp', # dp or fsdp.
    't_sampling': 'discrete-dt',
    'dt_sampling': 'uniform',
    'bootstrap_cfg': 0,
    'bootstrap_every': 8, # Make sure its a divisor of batch size.
    'bootstrap_ema': 1,
    'bootstrap_dt_bias': 0,
    'train_type': 'shortcut_anchor', # Use anchor-based locality loss
    # Anchor-specific parameters
    'use_label_anchors': True,  # Use class-based anchors (simpler) vs learned centroids
    'anchor_weight': 1.0,  # Weight for anchor consistency loss
    'use_cluster_centroids': False,  # Use precomputed cluster centroids from encode_and_cluster.py
    'num_clusters': 100,  # Number of clusters (if using cluster centroids)
})


wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'shortcut-anchor',
    'name': 'anchor_{dataset_name}',
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
    if jax.process_index() == 0 and FLAGS.mode == 'train':
        # Ensure we pass a plain dict (not FieldReference/ConfigDict wrappers)
        setup_wandb(FLAGS.model.to_dict(), **(FLAGS.wandb.to_dict() if hasattr(FLAGS.wandb, 'to_dict') else FLAGS.wandb))
        
    dataset = get_dataset(FLAGS.dataset_name, local_batch_size, True, FLAGS.debug_overfit)
    dataset_valid = get_dataset(FLAGS.dataset_name, local_batch_size, False, FLAGS.debug_overfit)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[:1]
    example_obs_shape = example_obs.shape

    # Load cluster centroids if using precomputed clusters
    cluster_centroids = None
    if FLAGS.model.use_cluster_centroids and FLAGS.cluster_dir is not None:
        cluster_path = Path(FLAGS.cluster_dir) / "cluster_centers.npy"
        if cluster_path.exists():
            cluster_centroids = np.load(cluster_path)
            print(f"Loaded {cluster_centroids.shape[0]} cluster centroids from {cluster_path}")
            FLAGS.model.num_clusters = cluster_centroids.shape[0]
            # Convert to JAX array and decode if needed
            cluster_centroids = jnp.array(cluster_centroids)
        else:
            print(f"Warning: cluster_centers.npy not found at {cluster_path}, falling back to label-based anchors")
            FLAGS.model.use_cluster_centroids = False

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
        
        # If using cluster centroids, decode them to latent space
        if cluster_centroids is not None and FLAGS.model.use_stable_vae:
            # Centroids are in latent space already if encoded with stablevae
            # Reshape to match latent dimensions
            latent_shape = example_obs_shape[1:]  # (h, w, c)
            if cluster_centroids.shape[1] == np.prod(latent_shape):
                cluster_centroids = cluster_centroids.reshape(-1, *latent_shape)
                print(f"Reshaped cluster centroids to {cluster_centroids.shape}")

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
        'ignore_dt': False,  # Anchor method uses dt like shortcut
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
        train_state = train_state.replace(**replace_dict)
        start_step = train_state.step
        train_state = jax.jit(lambda x : x, out_shardings=train_state_sharding)(train_state)
        print("Loaded model with step", train_state.step)
        jax.debug.visualize_array_sharding(train_state.params['FinalLayer_0']['Dense_0']['kernel'])
        del cp

    train_state_teacher = None

    visualize_labels = example_labels
    visualize_labels = shard_data(visualize_labels)
    visualize_labels = jax.experimental.multihost_utils.process_allgather(visualize_labels)
    imagenet_labels = open('data/imagenet_labels.txt').read().splitlines()

    # Shard cluster centroids if available
    if cluster_centroids is not None:
        cluster_centroids = shard_data(cluster_centroids)
        print(f"Sharded cluster centroids: {cluster_centroids.shape}")

    ###################################
    # Update Function
    ###################################

    @partial(jax.jit, out_shardings=(train_state_sharding, no_shard))
    def update(train_state, images, labels, centroids=None, force_t=-1, force_dt=-1):
        new_rng, targets_key, dropout_key, perm_key = jax.random.split(train_state.rng, 4)
        info = {}

        id_perm = jax.random.permutation(perm_key, images.shape[0])
        images = images[id_perm]
        labels = labels[id_perm]
        images = jax.lax.with_sharding_constraint(images, data_sharding)
        labels = jax.lax.with_sharding_constraint(labels, data_sharding)

        if FLAGS.model['cfg_scale'] == 0: # For unconditional generation.
            labels = jnp.ones(labels.shape[0], dtype=jnp.int32) * FLAGS.model['num_classes']

        # Import and use anchor-based targets
        from targets_shortcut_LFM_anchor import get_targets
        x_t, v_t, t, dt_base, labels_dropped, info = get_targets(
            FLAGS, targets_key, train_state, images, labels, force_t, force_dt, 
            cluster_centroids=centroids
        )

        def loss_fn(grad_params):
            v_prime, logvars, activations = train_state.call_model(x_t, t, dt_base, labels_dropped, train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True)
            mse_v = jnp.mean((v_prime - v_t) ** 2, axis=(1, 2, 3))
            loss = jnp.mean(mse_v)

            loss_info = {
                'loss': loss,
                'v_magnitude_prime': jnp.sqrt(jnp.mean(jnp.square(v_prime))),
                **{'activations/' + k : jnp.sqrt(jnp.mean(jnp.square(v))) for k, v in activations.items()},
            }

            # Add anchor consistency loss
            # Check if anchor information is available in info dict
            if 'anchor_xt' in info and 'anchor_v' in info:
                anchor_xt = info['anchor_xt']
                anchor_v = info['anchor_v']
                anchor_weight = info['anchor_weight']
                anchor_valid_mask = info['anchor_valid_mask']
                
                # Get number of bootstrap samples
                bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
                bst_size_data = FLAGS.batch_size - bootstrap_size
                
                # Only apply anchor loss to non-bootstrap samples
                if bst_size_data > 0:
                    # Extract non-bootstrap portion
                    anchor_xt_flow = anchor_xt[:bst_size_data]
                    t_flow = t[bootstrap_size:]
                    dt_base_flow = dt_base[bootstrap_size:]
                    labels_flow = labels_dropped[bootstrap_size:]
                    anchor_valid_flow = anchor_valid_mask[:bst_size_data]
                    
                    # Predict velocity at anchor points
                    v_anchor, _, _ = train_state.call_model(
                        anchor_xt_flow, t_flow, dt_base_flow, labels_flow, 
                        train=True, rngs={'dropout': dropout_key}, params=grad_params, return_activations=True
                    )
                    
                    # Predict velocity at original points (flow portion only)
                    v_original_flow = v_prime[bootstrap_size:]
                    
                    # Compute anchor consistency loss: ||v(x_t, t) - v(anchor_xt, t)||²
                    anchor_mse = jnp.mean((v_original_flow - v_anchor) ** 2, axis=(1, 2, 3))
                    # Apply valid mask (only for valid labels/anchors)
                    anchor_mse_masked = jnp.where(anchor_valid_flow, anchor_mse, 0.0)
                    anchor_loss = jnp.sum(anchor_mse_masked) / jnp.maximum(jnp.sum(anchor_valid_flow), 1.0)
                    
                    # Add to total loss
                    loss = loss + anchor_weight * anchor_loss
                    loss_info['loss_anchor'] = anchor_loss
                    loss_info['anchor_valid_ratio'] = jnp.mean(anchor_valid_flow.astype(jnp.float32))

            # Track bootstrap vs flow losses
            bootstrap_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
            loss_info['loss_flow'] = jnp.mean(mse_v[bootstrap_size:])
            loss_info['loss_bootstrap'] = jnp.mean(mse_v[:bootstrap_size])
            # Record total loss after any additions (e.g., anchor loss).
            # Keep the original 'loss' key as the base MSE; expose the final
            # optimized objective under 'loss_total'.
            loss_info['loss_total'] = loss

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
        do_inference(FLAGS, train_state, None, dataset, dataset_valid, shard_data, vae_encode, vae_decode, 
                    lambda ts, tst, imgs, lbls: update(ts, imgs, lbls, cluster_centroids),
                    get_fid_activations, imagenet_labels, visualize_labels, 
                    fid_from_stats, truth_fid_stats)
        return

    ###################################
    # Train Loop
    ###################################

    try:
        for i in tqdm.tqdm(range(1 + start_step, FLAGS.max_steps + 1 + start_step),
                           smoothing=0.1,
                           dynamic_ncols=True):

            # Sample data.
            if not FLAGS.debug_overfit or i == 1:
                batch_images, batch_labels = shard_data(*next(dataset))
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    vae_rng, vae_key = jax.random.split(vae_rng)
                    batch_images = vae_encode(vae_key, batch_images)

            # Train update.
            train_state, update_info = update(train_state, batch_images, batch_labels, cluster_centroids)

            if i % FLAGS.log_interval == 0 or i == 1:
                update_info = jax.device_get(update_info)
                update_info = jax.tree_util.tree_map(lambda x: np.array(x), update_info)
                update_info = jax.tree_util.tree_map(lambda x: x.mean(), update_info)
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}

                valid_images, valid_labels = shard_data(*next(dataset_valid))
                if FLAGS.model.use_stable_vae and 'latent' not in FLAGS.dataset_name:
                    valid_images = vae_encode(vae_rng, valid_images)
                _, valid_update_info = update(train_state, valid_images, valid_labels, cluster_centroids)
                valid_update_info = jax.device_get(valid_update_info)
                valid_update_info = jax.tree_util.tree_map(lambda x: x.mean(), valid_update_info)
                train_metrics['training/loss_valid'] = valid_update_info['loss']

                if jax.process_index() == 0:
                    wandb.log(train_metrics, step=int(train_state.step.item()))

            if i % FLAGS.eval_interval == 0:
                eval_model(FLAGS, train_state, train_state_teacher, int(train_state.step.item()), dataset, dataset_valid, shard_data, vae_encode, vae_decode, 
                          lambda ts, tst, imgs, lbls: update(ts, imgs, lbls, cluster_centroids),
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
    finally:
        # Finish wandb run on the main process if it was initialized
        if jax.process_index() == 0 and FLAGS.mode == 'train':
            try:
                wandb.finish()
            except Exception:
                pass

if __name__ == '__main__':
    app.run(main)
