#! /usr/bin/bash

# uv run encode_and_cluster.py \
#   --dataset-name celebahq256 \
#   --n-samples 30000 \
#   --batch-size 8 \
#   --image-size 512 \
#   --encoder stablevae \
#   --out-dir results_celebahq \
#   --n-clusters 100 \
#   --cluster-method kmeans
uv run train.py \
  --model.hidden_size 768 \
  --model.patch_size 2 \
  --model.depth 12 \
  --model.num_heads 12 \
  --model.mlp_ratio 4 \
  --dataset_name celebahq256 \
  --fid_stats data/celeba256_fidstats_ours.npz \
  --model.cfg_scale 0 \
  --model.class_dropout_prob 1 \
  --model.num_classes 1 \
  --batch_size 64 \
  --max_steps 410_000 \
  --model.train_type shortcut \
  --max_steps 400001 \
  --eval_interval 5000 \
  --log_interval 5000 \
  --save_dir ./checkpoints_celebahq_base \
  --save_interval 35000 \