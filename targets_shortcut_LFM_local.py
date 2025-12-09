import jax
import jax.numpy as jnp
import numpy as np

def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1, cluster_assignments=None):
    """
    Locality-aware flow matching with bootstrap distillation.
    Implements L_local from Equation 4: enforces v(u_t, t) ≈ v(s_t, t) for similar samples.
    
    Two modes:
    1. Random perturbation (default): Two similar noises should produce two similar images.
    2. Cluster neighborhood (if cluster_assignments provided): Two samples from the same 
       cluster/neighborhood should have similar trajectories.
    
    Args:
        cluster_assignments: Optional array of cluster IDs for each sample in the batch.
                           Shape: (batch_size,). If provided, uses cluster-based pairing
                           instead of random perturbation.
    """
    label_key, time_key, noise_key, perturbation_key, pair_key = jax.random.split(key, 5)
    info = {}

    # 1) =========== Sample dt. ============
    bootstrap_batchsize = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    log2_sections = np.log2(FLAGS.model['denoise_timesteps']).astype(np.int32)
    if FLAGS.model['bootstrap_dt_bias'] == 0:
        dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections), bootstrap_batchsize // log2_sections)
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize-dt_base.shape[0],)])
        num_dt_cfg = bootstrap_batchsize // log2_sections
    else:
        dt_base = jnp.repeat(log2_sections - 1 - jnp.arange(log2_sections-2), (bootstrap_batchsize // 2) // log2_sections)
        dt_base = jnp.concatenate([dt_base, jnp.ones(bootstrap_batchsize // 4), jnp.zeros(bootstrap_batchsize // 4)])
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize-dt_base.shape[0],)])
        num_dt_cfg = (bootstrap_batchsize // 2) // log2_sections
    force_dt_vec = jnp.ones(bootstrap_batchsize, dtype=jnp.float32) * force_dt
    dt_base = jnp.where(force_dt_vec != -1, force_dt_vec, dt_base)
    dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/4, 1/8, 1/16, 1/32]
    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2

    # 2) =========== Sample t. ============
    dt_sections = jnp.power(2, dt_base) # [1, 2, 4, 8, 16, 32]
    t = jax.random.randint(time_key, (bootstrap_batchsize,), minval=0, maxval=dt_sections).astype(jnp.float32)
    t = t / dt_sections # Between 0 and 1.
    force_t_vec = jnp.ones(bootstrap_batchsize, dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]

    # 3) =========== Generate Bootstrap Targets ============
    x_1 = images[:bootstrap_batchsize]
    x_0 = jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    bst_labels = labels[:bootstrap_batchsize]
    call_model_fn = train_state.call_model if FLAGS.model['bootstrap_ema'] == 0 else train_state.call_model_ema
    if not FLAGS.model['bootstrap_cfg']:
        v_b1 = call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False)
        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)
        v_b2 = call_model_fn(x_t2, t2, dt_base_bootstrap, bst_labels, train=False)
        v_target = (v_b1 + v_b2) / 2
    else:
        x_t_extra = jnp.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
        t_extra = jnp.concatenate([t, t[:num_dt_cfg]], axis=0)
        dt_base_extra = jnp.concatenate([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], axis=0)
        labels_extra = jnp.concatenate([bst_labels, jnp.ones(num_dt_cfg, dtype=jnp.int32) * FLAGS.model['num_classes']], axis=0)
        v_b1_raw = call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False)
        v_b_cond = v_b1_raw[:x_1.shape[0]]
        v_b_uncond = v_b1_raw[x_1.shape[0]:]
        v_cfg = v_b_uncond + FLAGS.model['cfg_scale'] * (v_b_cond[:num_dt_cfg] - v_b_uncond)
        v_b1 = jnp.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

        t2 = t + dt_bootstrap
        x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
        x_t2 = jnp.clip(x_t2, -4, 4)
        x_t2_extra = jnp.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
        t2_extra = jnp.concatenate([t2, t2[:num_dt_cfg]], axis=0)
        v_b2_raw = call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False)
        v_b2_cond = v_b2_raw[:x_1.shape[0]]
        v_b2_uncond = v_b2_raw[x_1.shape[0]:]
        v_b2_cfg = v_b2_uncond + FLAGS.model['cfg_scale'] * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = jnp.concatenate([v_b2_cfg, v_b2_cond[num_dt_cfg:]], axis=0)
        v_target = (v_b1 + v_b2) / 2

    v_target = jnp.clip(v_target, -4, 4)
    bst_v = v_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t
    bst_l = bst_labels

    # 4) =========== Generate Locality-Aware Flow-Matching Targets ============
    # Implements: L_local = E[ℓ(t, x_0, x_1)] + λ E[||v(u_t, t) - v(s_t, t)||²]
    # Two modes:
    # - Random perturbation: u_t and s_t come from similar noises (x_0 and x_0')
    # - Cluster neighborhood: u_t and s_t come from samples in the same cluster

    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # Sample t.
    t = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=FLAGS.model['denoise_timesteps']).astype(jnp.float32)
    t /= FLAGS.model['denoise_timesteps']
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)         # If force_t is not -1, then use force_t.
    t_full = t[:, None, None, None] # [batch, 1, 1, 1]

    # Sample flow pairs x_t, v_t.
    x_0 = jax.random.normal(noise_key, images.shape)
    x_1 = images
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0
    
    # ==== Choose locality pairing strategy ====
    use_cluster_neighborhoods = FLAGS.model.get('use_cluster_neighborhoods', False)
    
    if use_cluster_neighborhoods and cluster_assignments is not None:
        # Mode 1: Cluster-based pairing
        # For each sample, find another sample from the same cluster in the batch
        # This creates semantically meaningful locality constraints
        
        batch_clusters = cluster_assignments
        batch_size = images.shape[0]
        
        # Create pairing: for each sample i, find a partner j from same cluster
        # Strategy: randomly shuffle indices within each cluster
        pair_indices = jnp.arange(batch_size)
        
        # For each unique cluster, create random pairings within cluster
        unique_clusters = jnp.unique(batch_clusters)
        
        # Simple pairing strategy: for each sample, find next sample in same cluster (circular)
        # This ensures every sample has a pair from its cluster
        pair_indices_list = []
        valid_pairs = []
        
        for cluster_id in range(jnp.max(batch_clusters) + 1):
            # Find all samples in this cluster
            cluster_mask = (batch_clusters == cluster_id)
            cluster_indices = jnp.where(cluster_mask, jnp.arange(batch_size), -1)
            cluster_indices = cluster_indices[cluster_indices >= 0]
            n_in_cluster = jnp.sum(cluster_mask)
            
            if n_in_cluster > 1:
                # Randomly shuffle within cluster to create pairs
                shuffled = jax.random.permutation(pair_key, cluster_indices)
                # Map each original index to its shuffled partner
                for idx, orig_idx in enumerate(cluster_indices):
                    pair_indices_list.append((orig_idx, shuffled[idx]))
                    valid_pairs.append(True)
            else:
                # Singleton cluster - pair with self (will be marked invalid)
                if n_in_cluster == 1:
                    orig_idx = cluster_indices[0]
                    pair_indices_list.append((orig_idx, orig_idx))
                    valid_pairs.append(False)
        
        # Convert to arrays - use a simpler approach for JAX compatibility
        # Pair each sample with a random other sample from same cluster
        pair_indices = jnp.zeros(batch_size, dtype=jnp.int32)
        valid_pair_mask = jnp.zeros(batch_size, dtype=bool)
        
        for i in range(batch_size):
            # Find all samples in same cluster as sample i
            same_cluster = (batch_clusters == batch_clusters[i])
            same_cluster_indices = jnp.where(same_cluster, jnp.arange(batch_size), batch_size)
            # Exclude self
            same_cluster_indices = jnp.where(same_cluster_indices == i, batch_size, same_cluster_indices)
            # Count valid partners
            n_partners = jnp.sum(same_cluster_indices < batch_size)
            
            # If there are other samples in same cluster, pick one randomly
            if n_partners > 0:
                # Random selection from cluster-mates
                rand_idx = jax.random.randint(pair_key, (), 0, batch_size)
                selected = same_cluster_indices[rand_idx % jnp.maximum(n_partners, 1)]
                pair_indices = pair_indices.at[i].set(jnp.where(n_partners > 0, selected, i))
                valid_pair_mask = valid_pair_mask.at[i].set(True)
            else:
                # No cluster-mates, pair with self (invalid)
                pair_indices = pair_indices.at[i].set(i)
                valid_pair_mask = valid_pair_mask.at[i].set(False)
        
        # Get paired samples
        x_1_paired = x_1[pair_indices]
        x_0_paired = x_0  # Use same noise for fair comparison
        
        # Compute u_t (original) and s_t (from cluster neighbor)
        u_t = x_t  # Original trajectory point
        s_t = (1 - (1 - 1e-5) * t_full) * x_0_paired + t_full * x_1_paired  # Neighbor's trajectory
        
        # Target velocities
        v_u = v_t  # Original velocity
        v_s = x_1_paired - (1 - 1e-5) * x_0_paired  # Neighbor's velocity
        
        info['locality_mode'] = 'cluster_neighborhood'
        info['locality_valid_pairs'] = jnp.mean(valid_pair_mask.astype(jnp.float32))
        info['locality_n_unique_clusters'] = jnp.unique(batch_clusters).shape[0]
        
    else:
        # Mode 2: Random perturbation (original approach)
        # Create pairs: (x_0, x_0') where x_0' is a small perturbation of x_0
        locality_scale = FLAGS.model.get('locality_noise_scale', 0.1)
        x_0_perturbed = x_0 + locality_scale * jax.random.normal(perturbation_key, x_0.shape)
        
        # Compute u_t and s_t: two nearby points from similar noises
        u_t = x_t  # Original point (same as x_t)
        s_t = (1 - (1 - 1e-5) * t_full) * x_0_perturbed + t_full * x_1  # Perturbed point
        
        # Both should have similar velocity targets (since they lead to same x_1)
        v_u = v_t
        v_s = x_1 - (1 - 1e-5) * x_0_perturbed
        
        valid_pair_mask = jnp.ones(images.shape[0], dtype=bool)  # All pairs valid
        info['locality_mode'] = 'random_perturbation'
    
    # Store locality information for loss computation
    # The model will predict v(u_t, t) and v(s_t, t), and we'll add a consistency term
    info['locality_u_t'] = u_t
    info['locality_s_t'] = s_t
    info['locality_v_u'] = v_u
    info['locality_v_s'] = v_s
    info['locality_weight'] = FLAGS.model.get('locality_weight', 1.0)
    info['locality_valid_mask'] = valid_pair_mask
    
    dt_flow = np.log2(FLAGS.model['denoise_timesteps']).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    # ==== 5) Merge Flow+Bootstrap ====
    bst_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    bst_size_data = FLAGS.batch_size - bst_size
    x_t = jnp.concatenate([bst_xt, x_t[:bst_size_data]], axis=0)
    t = jnp.concatenate([bst_t, t[:bst_size_data]], axis=0)
    dt_base = jnp.concatenate([bst_dt, dt_base[:bst_size_data]], axis=0)
    v_t = jnp.concatenate([bst_v, v_t[:bst_size_data]], axis=0)
    labels_dropped = jnp.concatenate([bst_l, labels_dropped[:bst_size_data]], axis=0)
    info['bootstrap_ratio'] = jnp.mean(dt_base != dt_flow)

    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(bst_v)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_t, t, dt_base, labels_dropped, info
