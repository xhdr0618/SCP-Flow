#!/bin/bash
echo "$(date +"[%Y-%m-%d %H:%M:%S]") start evaluate SCP-Flow"

eval_path=${1:-"results/test_scpflow"}

python metrics/evaluate_scpflow.py --eval_path "$eval_path" --batch_size 1 --num_worker 0

echo "$(date +"[%Y-%m-%d %H:%M:%S]") end evaluate SCP-Flow"
