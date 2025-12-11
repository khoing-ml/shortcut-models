#!/usr/bin/env bash
# Run Train Anchor with sensible defaults. Edit flags below as needed.
# Usage: ./train_anchor_run.sh [extra args]

set -euo pipefail

# Example environment variables you might want to set for multi-GPU/TPU runs:
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run encode_and_cluster.py \
  --dataset-name celebahq256 \
  --n-samples 30000 \
  --batch-size 8 \
  --image-size 256 \
  --encoder stablevae \
  --out-dir results_celebahq \
  --n-clusters 1024 \
  --cluster-method kmeans
PYTHON=python3
SCRIPT=./train_anchor.py

# Default args (edit to taste)
ARGS=(
  --dataset_name celebahq256
  --cluster_dir ./results_celebahq
  --model.hidden_size 768
  --model.patch_size 2
  --model.depth 12
  --model.num_heads 12
  --model.mlp_ratio 4
  --model.cfg_scale 0
  --model.class_dropout_prob 1
  --model.num_classes 1
  --model.train_type shortcut
  --model.sharding fsdp
  --model.bootstrap_every 4
  --model.num_clusters 1024
  --batch_size 64
  --fid_stats data/celeba256_fidstats_ours.npz
  --max_steps 400001
  --eval_interval 5000
  --log_interval 5000
  --save_dir ./checkpoints_celebahq_anchor
  --save_interval 35000
  --model.use_cluster_centroids=True
)

# Allow passing extra args on the command line to override or extend
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi

echo "Running: $PYTHON $SCRIPT ${ARGS[*]} ${EXTRA_ARGS[*]}"
exec "$PYTHON" "$SCRIPT" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
