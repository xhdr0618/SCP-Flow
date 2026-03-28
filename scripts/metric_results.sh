#!/bin/bash
echo "$(date +"[%Y-%m-%d %H:%M:%S]") start test LAtMSHFPMQM"
eval_path="result/test_LAtMSHFPMQM"  # path to your generation results (image/(*_gen.png | *_gt.png))
# Visual evaluation
python metrics/calculate_metric.py --eval_path $eval_path --suffix_gt _gt.png --suffix_gene _gen.png

# Category evaluation
python metrics/calculate_metric_class.py --eval_path $eval_path  --info_path 'data/SIGF_info.xlsx' --classifier_ckpt_path 'pre-trained/image_classifier/classifier.ckpt'
echo "$(date +"[%Y-%m-%d %H:%M:%S]") end test LAtMSHFPMQM"
