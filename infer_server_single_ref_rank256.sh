#!/usr/bin/env bash

# ===== Environment =====
conda activate reco

# ===== Shared args =====
CKPT_1="all_ckpts/ReCo_ref_rank256-2026_m4_version.ckpt"

# run each_task 
# ================ Setting 1: prompt + IP image======================
# IP image only, no first frame
video_path="assets\test_videos\replace_ref_ori.mp4"
prompt="Replace the white and black robot with glowing green eyes in the foreground with a woman with brown hair and a floral top."
ip_img_path="assets\test_videos\replace_ref_ip.png"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt --ip_img_path $ip_img_path

video_path="assets\test_videos\local_change_ref_ori.mp4"
prompt="Replace the man's black formal suit with a vibrant red tuxedo and black bow tie, ensuring it fits his pose and position within the scene."
ip_img_path="assets\test_videos\local_change_ip.png"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt --ip_img_path $ip_img_path

video_path="assets\test_videos\background_change_ref_ori_2.mp4"
prompt="Replace the dynamic futuristic newsroom with the original static indoor atrium scene, characterized by steady lighting, people walking calmly, and no digital effects."
ip_img_path="assets\test_videos\background_change_ref_ip_2.png"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt --ip_img_path $ip_img_path

video_path="assets\test_videos\background_change_ref_ori.mp4"
prompt="In this video clip, replace the modern gourmet kitchen with the original refrigerator setting, ensuring the subject's position and appearance remain unchanged."
ip_img_path="assets\test_videos\background_change_ref_ip.png"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt --ip_img_path $ip_img_path

# Provide Both..Optional
# # ================ Setting 2: prompt + IP image + first frame======================
# python inference_reco_single_ref.py --video_path video.mp4 --prompt "..." \
#     --first_frame_path frame.png --ip_img_path ip.png


# ================ Setting 3: Prompt only =============
prompt="Add a mallard duck swimming to the left of the male duck."
video_path="assets/test_videos/1000734-hd_1920_1080_24fps.mp4"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt

prompt="Remove the spotted baby seal from the beach."
video_path="assets/test_videos/1526909-hd_1920_1080_24fps.mp4"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt

prompt="Replace the mallard duck with a brown muskrat swimming in the water."
video_path="assets/test_videos/1500734-hd_1920_1080_24fps.mp4"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt

prompt="Cast it as the 3D Chibi style."
video_path="assets/test_videos/3129424-uhd_3840_2160_24fps.mp4"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt    


# ================ Setting 4: First frame only =============
video_path="assets/test_videos/1526909-hd_1920_1080_24fps.mp4"
prompt="Remove the spotted baby seal from the beach."
first_frame_path="assets/test_videos/1500734-hd_1920_1080_24fps_edited_frame0.png"
python inference_reco_single_ref.py --video_path $video_path --prompt $prompt --first_frame_path $first_frame_path

