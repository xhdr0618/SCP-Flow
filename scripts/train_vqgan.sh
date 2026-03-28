#!/bin/bash
mkdir ./result_VQGAN
echo "$(date +"[%Y-%m-%d %H:%M:%S]") start train image VQGAN"
python train_vqgan.py --command "fit" --batch_size 16 --num_workers 18 --result_root "result_VQGAN" --data_root "path_to_your_fundus_image_dataset"
echo "$(date +"[%Y-%m-%d %H:%M:%S]") end train image VQGAN"
