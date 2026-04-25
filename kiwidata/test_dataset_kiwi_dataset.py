import torch
from torch.utils.data import DataLoader
import torch
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_dataset_unify import UnifiedDataset_kiwidata, collate_fn_with_diff_mask
import numpy as np
import shutil
import imageio


def save_video(frames, save_path, fps, quality=5, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()



class SimpleArgs:
    dataset_folder_dict = {
        "Ditto-1M": "s3://hidream-dataset-ditto1m/Ditto-1M/videos",
        "openve3m": "s3://hidream-user-zhangzhongwei/Dataset/OpenVE-3M/videos_folders",
        "reco_data": "s3://hidream-dataset-opens2v5m/OpenS2V-processed/Editdata/ReCo_Data_4",
    }
    ref_img_folder="/mnt/zhongwei/Data/linyq/kiwi_edit_training_data/refvie_477k_stage3/ref_images"
    vid_ref_dataset_metadata_path = "/mnt/zhongwei/Data/linyq/kiwi_edit_training_data/refvie_dataset.csv"
    data_file_keys = "src_video,tgt_video,ref_image,prompt"
    num_frames = 81
    height = 480
    width = 832

args = SimpleArgs()

rank=0
world_size=1
vid_ref_dataset = UnifiedDataset_kiwidata(
    dataset_folder_dict=args.dataset_folder_dict,
    ref_img_folder=args.ref_img_folder,
    metadata_path=args.vid_ref_dataset_metadata_path,
    data_file_keys=args.data_file_keys.split(","),
    rank=rank, world_size=world_size,
    shuffle=True,
)
vid_ref_dataloader = DataLoader(vid_ref_dataset, shuffle=False, collate_fn=collate_fn_with_diff_mask, num_workers=0)


# 4. 简单迭代看一眼
for i, batch in enumerate(vid_ref_dataloader):
    print(f"\n--- 第 {i+1} 个样本 ---")
    
    if isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"Key: {key:15} | Shape: {list(value.shape)} | Dtype: {value.dtype} | Scope: {value.max(), value.min()}")
            else:
                print(f"Key: {key:15} | Value Type: {type(value)}")
    else:
        print(f"Batch 类型不是字典，是: {type(batch)}")

    # ========== Start iter ==========
    video_name = batch['video_name'][0]
    video_save_path = os.path.join('all_cached/video_cache', f'{video_name}.mp4')
    tar_video = (batch['tar_video'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255

    tar_video_list = [tar_video[i].astype(np.uint8) for i in range(tar_video.shape[0])]
    os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
    save_video(tar_video_list, video_save_path, fps=16, quality=5)

    # video_save_path = os.path.join('all_cached/video_cache', task_name, f'{video_name}_diff_mask.mp4')
    # diff_mask = data_iter['diff_mask'][0].permute(1,2,3,0).numpy()*255

    # diff_mask_list = [diff_mask[i].astype(np.uint8) for i in range(diff_mask.shape[0])]
    # os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
    # save_video(diff_mask_list, video_save_path, fps=16, quality=5)

    # pil image save
    pil_img_path = batch['ref_img_path'][0]
    img_save_path = video_save_path.replace('.mp4', '.png')
    if pil_img_path is not None:
        shutil.copy(pil_img_path, img_save_path)


    prompt = batch['prompt'][0]
    txt_save_path = video_save_path.replace('.mp4', '.txt')
    with open(txt_save_path, 'w') as f:
        f.write(str(prompt))


    if i>=60:
        break



