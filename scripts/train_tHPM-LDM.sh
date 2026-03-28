#!/bin/bash
mkdir ./result
mkdir ./result/train_LAtMSHFPMQM
echo "$(date +"[%Y-%m-%d %H:%M:%S]") start train LAtMSHFPMQM"
python train_vqldm_PMN.py --command "fit" --exp_name "LAtMSHFPMQM" --max_epochs 700 --batch_size 20 --num_workers 4 --accumulate_grad_batches 1 --base_learning_rate 5.0e-05 --data_root "data/SIGF_make" --result_root "result/train_LAtMSHFPMQM" --first_stage_ckpt "pre-trained/VQGAN/vqgan.ckpt" --diff_pred_cond "LAtMSHFPMQM" --diff_time 50
echo "$(date +"[%Y-%m-%d %H:%M:%S]") end train LAtMSHFPMQM"
