#!/usr/bin/env bash

# ===== 环境 =====
conda activate reco

# ===== 公共参数 =====
CKPT_1="all_ckpts/ReCo_ori_rank128-2025_m12_version.ckpt"

# run each_task 
python inference_reco_single.py --test_txt_file_name assets/replace_test.txt --task_name replace --lora_ckpt $CKPT_1
python inference_reco_single.py --test_txt_file_name assets/remove_test.txt --task_name remove --lora_ckpt $CKPT_1
python inference_reco_single.py --test_txt_file_name assets/style_test.txt --task_name style --lora_ckpt $CKPT_1
python inference_reco_single.py --test_txt_file_name assets/add_test.txt --task_name add --lora_ckpt $CKPT_1

# run Propagation: each_task with first frame condition... 
python inference_reco_single.py --test_txt_file_name assets/remove_test_given_first_frame.txt --task_name remove --lora_ckpt $CKPT_1


