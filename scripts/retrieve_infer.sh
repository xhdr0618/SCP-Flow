mkdir ./results_retrieve

eval_dir="./result_retrieve/PMQM-retrieve-test"
resume="pre-trained/tHPM-LDM/tHPM-LDM.ckpt"
mkdir $eval_dir
python retrieve_vqldm_PMN.py --command "test" --batch_size 9 --num_workers 3 --data_root "data/SIGF_make" --image_save_dir "$eval_dir" --first_stage_ckpt "pre-trained/VQGAN/vqgan.ckpt" --resume "$resume" --test_type 'test' --diff_time 1

eval_dir="./result_retrieve/PMQM-retrieve-train"
resume="pre-trained/tHPM-LDM/tHPM-LDM.ckpt"
mkdir $eval_dir
python retrieve_vqldm_PMN.py --command "test" --batch_size 6 --num_workers 3 --data_root "data/SIGF_make" --image_save_dir "$eval_dir" --first_stage_ckpt "pre-trained/VQGAN/vqgan.ckpt" --resume "$resume" --test_type 'train' --diff_time 1