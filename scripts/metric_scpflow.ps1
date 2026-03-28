$ErrorActionPreference = "Stop"

param(
    [string]$EvalPath = "results/test_scpflow"
)

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start evaluate SCP-Flow"

python metrics/evaluate_scpflow.py --eval_path $EvalPath --batch_size 1 --num_worker 0

Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] end evaluate SCP-Flow"
