#!/bin/bash
mkdir -p ./results
mkdir -p ./results/train_scpflow
echo "$(date +"[%Y-%m-%d %H:%M:%S]") start train SCP-Flow"

python train_vqldm_PMN.py \
  --predictor_type "scpflow" \
  --command "fit" \
  --exp_name "SCPFlow" \
  --max_epochs 200 \
  --batch_size 8 \
  --num_workers 4 \
  --accumulate_grad_batches 1 \
  --base_learning_rate 5.0e-05 \
  --data_root "data/SIGF_make" \
  --result_root "results/train_scpflow" \
  --first_stage_ckpt "pre-trained/VQGAN/vqgan.ckpt" \
  --condition_dim 128 \
  --flow_hidden_channels 128 \
  --flow_tau_embed_dim 128 \
  --flow_bridge_sigma 0.05 \
  --flow_velocity_weight 1.0 \
  --flow_target_weight 1.0 \
  --flow_recon_weight 1.0 \
  --interval_loss_weight 0.2 \
  --uncertainty_loss_weight 0.1 \
  --consistency_loss_weight 0.1 \
  --structure_loss_weight 0.05 \
  --structure_dim 128 \
  --structure_roi_size 96 \
  --missing_visit_count 0 \
  --missing_visit_strategy "none"

echo "$(date +"[%Y-%m-%d %H:%M:%S]") end train SCP-Flow"
