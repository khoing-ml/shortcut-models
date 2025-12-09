# Locality-Aware Flow Matching Training Scripts

This directory contains implementations of locality-aware losses for flow matching with bootstrap distillation, based on the research paper on flow matching with locality constraints.

## Overview

Three new training scripts implement different locality-aware objectives:

1. **`train_anchor.py`** - Anchor-based flow matching (Equation 6)
2. **`train_local.py`** - Locality-aware flow matching (Equation 4)
3. Target generation files:
   - **`targets_shortcut_LFM_anchor.py`** - Generates anchor-based targets
   - **`targets_shortcut_LFM_local.py`** - Generates locality-based targets

## Theory

### Anchor-based Loss (L_anchor)

From Equation 6 in the paper:
```
L_anchor = E[ℓ(t, x_0, x_1)] + λ ||v(x_t, t) - v(e(x_1)_t, t)||²
```

where `e(x_1)` is the nearest centroid/anchor to `x_1`.

**Key idea:** The trajectory of an image should be similar to that of its cluster centroid, ensuring straighter generation paths.

### Locality-aware Loss (L_local)

From Equation 4 in the paper:
```
L_local = E[ℓ(t, x_0, x_1)] + λ E[||v(u_t, t) - v(s_t, t)||²]
```

where `u_t` and `s_t` are trajectories from similar noises.

**Key idea:** Two samples from nearby noises should have similar velocity fields, making the flow robust to noise perturbations.

## Usage

### 1. Anchor-based Training

#### Option A: Using Class Labels as Anchors (Simple)

```bash
python train_anchor.py \
  --dataset_name imagenet256 \
  --batch_size 128 \
  --save_dir ./checkpoints/anchor \
  --model.use_label_anchors=True \
  --model.anchor_weight=1.0
```

This uses class-conditional batch statistics as proxy centroids.

#### Option B: Using Precomputed Cluster Centroids (Advanced)

First, encode your dataset and cluster it:

```bash
# Encode and cluster the dataset
python encode_and_cluster.py \
  --dataset-name imagenet256 \
  --n-samples 50000 \
  --n-clusters 100 \
  --out-dir ./cluster_results \
  --encoder stablevae
```

This creates:
- `cluster_results/cluster_centers.npy` - Centroid embeddings
- `cluster_results/latents.npy` - Encoded latents
- `cluster_results/assignments.csv` - Cluster assignments

Then train with these clusters:

```bash
python train_anchor.py \
  --dataset_name imagenet256 \
  --batch_size 128 \
  --save_dir ./checkpoints/anchor_clustered \
  --cluster_dir ./cluster_results \
  --model.use_cluster_centroids=True \
  --model.use_label_anchors=False \
  --model.anchor_weight=1.0 \
  --model.num_clusters=100
```

### 2. Locality-aware Training

```bash
python train_local.py \
  --dataset_name imagenet256 \
  --batch_size 128 \
  --save_dir ./checkpoints/local \
  --model.locality_noise_scale=0.1 \
  --model.locality_weight=1.0
```

## Key Parameters

### Anchor-based (`train_anchor.py`)

- `--model.use_label_anchors`: Use class labels as anchors (default: True)
- `--model.use_cluster_centroids`: Use precomputed clusters (default: False)
- `--model.anchor_weight`: Weight for anchor consistency loss (default: 1.0)
- `--model.num_clusters`: Number of clusters when using centroids (default: 100)
- `--cluster_dir`: Directory containing `cluster_centers.npy`

### Locality-aware (`train_local.py`)

- `--model.locality_noise_scale`: Magnitude of noise perturbation (default: 0.1)
- `--model.locality_weight`: Weight for locality consistency loss (default: 1.0)

### Common Parameters (Both Scripts)

- `--model.bootstrap_every`: Bootstrap frequency (default: 8)
- `--model.bootstrap_cfg`: Use CFG in bootstrap (default: 0)
- `--model.bootstrap_ema`: Use EMA model for bootstrap (default: 1)
- `--model.cfg_scale`: Classifier-free guidance scale (default: 4.0)

## Implementation Details

### Anchor-based Method

1. **Class-based anchors** (simple):
   - Computes per-class batch means as centroids
   - Each sample uses its class centroid as anchor
   - Fast but may not capture intra-class diversity

2. **Cluster-based anchors** (advanced):
   - Uses k-means clusters in latent space
   - Each sample assigned to nearest centroid
   - Better captures data structure but requires preprocessing

3. **Loss computation**:
   - Standard flow matching loss on `(x_t, v_t)` pairs
   - Additional loss enforcing `||v(x_t, t) - v(anchor_xt, t)||²`
   - Only applied to non-bootstrap samples

### Locality-aware Method

1. **Perturbation generation**:
   - For each noise `x_0`, creates perturbed `x_0'`
   - Both lead to same target `x_1`
   - Perturbation scale controlled by `locality_noise_scale`

2. **Loss computation**:
   - Standard flow matching loss
   - Additional loss enforcing `||v(u_t, t) - v(s_t, t)||²`
   - Encourages smooth/robust velocity field

### Bootstrap Distillation

Both methods preserve the bootstrap distillation mechanism:
- A portion of batch uses self-distilled multi-step targets
- Remaining portion uses locality-aware objectives
- Bootstrap ratio: `1 / bootstrap_every`

## Monitoring Training

Both scripts log to Weights & Biases:

**Anchor-based metrics:**
- `training/loss_anchor` - Anchor consistency loss
- `training/anchor_valid_ratio` - Fraction of valid anchors
- `training/cluster_distances` - Mean distance to centroids (cluster mode)
- `training/unique_clusters_used` - Number of active clusters per batch

**Locality-aware metrics:**
- `training/loss_locality` - Locality consistency loss
- `training/locality_target_diff` - Target velocity difference
- `training/v_magnitude_prime` - Predicted velocity magnitude

**Common metrics:**
- `training/loss_flow` - Flow matching loss
- `training/loss_bootstrap` - Bootstrap distillation loss
- `training/grad_norm` - Gradient norm

## File Structure

```
shortcut-models/
├── train_anchor.py                    # Anchor-based training script
├── train_local.py                     # Locality-aware training script
├── targets_shortcut_LFM_anchor.py     # Anchor target generation
├── targets_shortcut_LFM_local.py      # Locality target generation
├── encode_and_cluster.py              # Dataset encoding & clustering
└── LOCALITY_TRAINING.md               # This file
```

## Differences from Original train.py

1. **Target generation**: Uses new locality-aware target functions
2. **Loss function**: Adds auxiliary consistency losses
3. **Cluster loading**: `train_anchor.py` can load precomputed centroids
4. **WandB project names**: Separate projects for each method

## Tips

1. **Start with small weights**: Begin with `anchor_weight=0.5` or `locality_weight=0.5` and increase gradually
2. **Noise scale tuning**: For locality, try `locality_noise_scale` in [0.05, 0.2]
3. **Cluster preprocessing**: Run `encode_and_cluster.py` overnight for large datasets
4. **Memory usage**: Cluster centroids add minimal memory overhead
5. **Convergence**: Locality losses may slow initial training but improve final quality

## Expected Results

- **Straighter trajectories**: Both methods should produce more linear generation paths
- **Better few-step quality**: Improved sample quality with fewer sampling steps
- **Robust generation**: More stable outputs across different noise seeds
- **Class coherence**: Anchor method improves within-class consistency

## Citation

Based on the flow matching with locality constraints research (see attached PDF).
