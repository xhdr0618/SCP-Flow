$ErrorActionPreference = "Stop"

param(
    [string]$CheckpointPath = "results/train_scpflow/SCPFlow/lightning_logs/version_0/checkpoints/best_model-flow-epoch=000.ckpt"
)

New-Item -ItemType Directory -Force -Path "results" | Out-Null
New-Item -ItemType Directory -Force -Path "results/test_scpflow" | Out-Null

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start test SCP-Flow"

python train_vqldm_PMN.py `
  --predictor_type scpflow `
  --command test `
  --batch_size 1 `
  --num_workers 1 `
  --data_root data/SIGF_make `
  --image_save_dir results/test_scpflow `
  --first_stage_ckpt pre-trained/VQGAN/vqgan.ckpt `
  --resume $CheckpointPath `
  --test_type test `
  --structure_loss_weight 0.05 `
  --structure_dim 128 `
  --structure_roi_size 96 `
  --missing_visit_count 0 `
  --missing_visit_strategy none

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] end test SCP-Flow"
