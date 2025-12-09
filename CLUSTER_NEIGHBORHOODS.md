# Cluster Neighborhood Locality Training

## My Understanding of Your Idea

### Problem with Random Perturbation Approach
The original `train_local.py` implementation enforces locality by:
1. Taking a noise sample `x_0`
2. Creating a **randomly** perturbed version `x_0' = x_0 + ε·noise`
3. Both lead to the same target `x_1`
4. Enforcing `||v(u_t, t) - v(s_t, t)||² ≈ 0`

**Issue**: The perturbation is **artificial and arbitrary** - there's no semantic meaning to why these two points should have similar trajectories.

### Your Enhancement: Cluster-Based Neighborhoods

Instead of random perturbations, use **semantic neighborhoods** defined by data clustering:

1. **Define neighborhoods via clusters**: 
   - Use precomputed k-means clusters from `encode_and_cluster.py`
   - Each cluster represents a region of **semantically similar** data
   - Example: Cluster 5 might contain "all golden retriever images"

2. **Sample pairs from same neighborhood**:
   - For each sample `x_1^(i)` in batch, find its cluster (e.g., cluster 5)
   - Find another sample `x_1^(j)` in the **same cluster** within the batch
   - These are **true neighbors** in data space, not artificial perturbations

3. **Enforce locality within neighborhoods**:
   - Two samples from cluster 5 should have similar generation trajectories
   - Their velocities `v(x_t^(i), t)` and `v(x_t^(j), t)` should be close
   - This means: "images from the same visual category should be generated via similar paths"

4. **Key Insight**:
   ```
   Random perturbation:  x_0  vs  x_0 + noise  (same x_1, different starting noise)
   Cluster neighborhood: x_1^(i)  vs  x_1^(j)  (different images, same semantic cluster)
   ```

## Implementation

### Mode 1: Random Perturbation (Original)
```python
# Same target, perturbed noise
x_0_perturbed = x_0 + 0.1 * random_noise
u_t = (1-t)·x_0 + t·x_1
s_t = (1-t)·x_0_perturbed + t·x_1
```

### Mode 2: Cluster Neighborhood (New)
```python
# Different targets from same cluster
x_1_neighbor = find_cluster_mate(x_1, cluster_id)
u_t = (1-t)·x_0 + t·x_1              # Original sample's trajectory
s_t = (1-t)·x_0 + t·x_1_neighbor      # Neighbor's trajectory (same noise!)
```

**Key difference**: We use the **same starting noise** `x_0` but **different targets** from the same cluster.

## Why This Is Better

### Semantic Meaning
- **Random**: "Slightly different noises leading to same image should have similar velocities"
  - Artificial constraint, unclear benefit
  
- **Cluster**: "Different images from same category should follow similar generation paths"
  - Natural constraint, encourages learning category-level structure

### Trajectory Straightness
- Forces the model to learn **consistent paths within each cluster**
- Images in cluster 5 all follow similar straight paths
- Reduces trajectory curvature and complexity

### Generalization
- Model learns cluster-level patterns, not just individual image paths
- Better few-shot generation quality
- More stable across different sampling strategies

## Usage Example

### Step 1: Encode and Cluster Dataset
```bash
python encode_and_cluster.py \
  --dataset-name imagenet256 \
  --n-samples 50000 \
  --n-clusters 500 \
  --out-dir ./imagenet_clusters \
  --encoder stablevae
```

Creates:
- `imagenet_clusters/cluster_centers.npy` - 500 centroids
- `imagenet_clusters/assignments.csv` - Maps each sample to cluster
- `imagenet_clusters/latents.npy` - Encoded representations

### Step 2: Train with Cluster Neighborhoods
```bash
python train_local.py \
  --dataset_name imagenet256 \
  --batch_size 128 \
  --cluster_dir ./imagenet_clusters \
  --model.use_cluster_neighborhoods=True \
  --model.locality_weight=1.0 \
  --model.num_clusters=500
```

### Step 3: Compare with Random Perturbation (Baseline)
```bash
python train_local.py \
  --dataset_name imagenet256 \
  --batch_size 128 \
  --model.use_cluster_neighborhoods=False \
  --model.locality_noise_scale=0.1 \
  --model.locality_weight=1.0
```

## What the Model Learns

### Random Perturbation Mode
```
Loss = MSE(v_pred, v_target) + λ·||v(x_t, t) - v(x_t + noise, t)||²
```
- Main loss: Predict correct velocity
- Locality: Be stable under small noise perturbations

### Cluster Neighborhood Mode
```
Loss = MSE(v_pred, v_target) + λ·||v(x_t^i, t) - v(x_t^j, t)||²
                                    where cluster(x_1^i) = cluster(x_1^j)
```
- Main loss: Predict correct velocity
- Locality: Images in same cluster follow similar paths

## Technical Details

### Pairing Strategy
For each sample in batch:
1. Look up its cluster ID
2. Find all other samples in same cluster
3. Randomly pair with one cluster-mate
4. If no cluster-mates in batch → mark as invalid pair

### Valid Pair Masking
```python
if n_cluster_mates > 0:
    pair_with_random_cluster_mate()
    valid_pair_mask[i] = True
else:
    pair_with_self()  # No loss contribution
    valid_pair_mask[i] = False
```

### Loss Computation
```python
locality_loss = mean(||v(u_t, t) - v(s_t, t)||² for valid pairs only)
```

Only pairs with cluster-mates contribute to loss.

## Expected Benefits

1. **Straighter Trajectories**: Cluster-level consistency → less curved paths
2. **Better Clustering**: Model learns to respect data structure
3. **Improved Quality**: Semantically meaningful constraints → better samples
4. **Fewer Steps**: Straighter paths → faster convergence during inference

## Monitored Metrics

### Cluster Mode
- `training/locality_mode`: "cluster_neighborhood" or "random_perturbation"
- `training/locality_valid_pairs`: Fraction of samples with cluster-mates
- `training/locality_n_unique_clusters`: Active clusters per batch
- `training/locality_valid_ratio`: Valid pair ratio in loss

### Both Modes
- `training/loss_locality`: Locality consistency loss
- `training/locality_target_diff`: How different the target velocities are

## Limitations & Considerations

1. **Batch Size**: Need sufficient batch size to have multiple samples per cluster
   - Recommended: batch_size ≥ 128 for 100-500 clusters
   
2. **Cluster Quality**: Effectiveness depends on clustering quality
   - Use appropriate number of clusters (too few → no specificity, too many → no pairs)
   
3. **Dataset Order**: Current implementation assumes sequential iteration
   - For shuffled datasets, may need to track image IDs
   
4. **Computational Cost**: Slightly higher (2x model calls per sample in locality loss)

## Summary

Your idea transforms locality from an **artificial noise robustness constraint** into a **semantic trajectory consistency constraint**. Instead of saying "be robust to noise perturbations," we now say "images from the same visual cluster should follow similar generation paths." This is more principled and should lead to better learned structure.
