#!/bin/bash
mkdir -p ./results
mkdir -p ./results/test_scpflow
echo "$(date +"[%Y-%m-%d %H:%M:%S]") start test SCP-Flow"

checkpoint_path=${1:-"results/train_scpflow/SCPFlow/lightning_logs/version_0/checkpoints/best_model-flow-epoch=000.ckpt"}

python train_vqldm_PMN.py \
  --predictor_type "scpflow" \
  --command "test" \
  --batch_size 1 \
  --num_workers 1 \
  --data_root "data/SIGF_make" \
  --image_save_dir "results/test_scpflow" \
  --first_stage_ckpt "pre-trained/VQGAN/vqgan.ckpt" \
  --resume "$checkpoint_path" \
  --test_type "test" \
  --structure_loss_weight 0.05 \
  --structure_dim 128 \
  --structure_roi_size 96 \
  --missing_visit_count 0 \
  --missing_visit_strategy "none"

echo "$(date +"[%Y-%m-%d %H:%M:%S]") end test SCP-Flow"
