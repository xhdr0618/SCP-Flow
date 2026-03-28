#!/bin/bash
mkdir ./result
mkdir ./result/test_LAtMSHFPMQM
echo "$(date +"[%Y-%m-%d %H:%M:%S]") start test LAtMSHFPMQM"
# data_root: path to your SIGF_make
python train_vqldm_PMN.py --command "test" --batch_size 1 --num_workers 1 --data_root "data/SIGF_make" --image_save_dir "result/test_LAtMSHFPMQM" --first_stage_ckpt "pre-trained/VQGAN/vqgan.ckpt" --resume "pre-trained/tHPM-LDM/tHPM-LDM.ckpt" --test_type 'test' --diff_time 1000 --diff_pred_cond "LAtMSHFPMQM"
echo "$(date +"[%Y-%m-%d %H:%M:%S]") end test LAtMSHFPMQM"