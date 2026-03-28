#!/bin/bash
mkdir ./result_classifier
echo "$(date +"[%Y-%m-%d %H:%M:%S]") start train image classifier"
python train_classifier.py --command "fit" --exp_name "image_classifier" --max_epochs 300 --batch_size 4 --num_workers 1 --accumulate_grad_batches 1 --base_learning_rate 3.0e-5 --loss_weight "(0.64, 0.36)" --data_root "your_classification_image_database" --result_root "./result_classifier"
echo "$(date +"[%Y-%m-%d %H:%M:%S]") end train classifier"
