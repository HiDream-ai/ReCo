import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import time
import contextlib
import uuid
import decord
import boto3
from torch.utils.data import DataLoader
import imageio
from tqdm import tqdm
import torch.nn.functional as F
import random


@contextlib.contextmanager
def temp_file_context_manager(directory, suffix):
    filename = os.path.join(directory, f"{uuid.uuid4()}{suffix}")
    try:
        yield filename
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def download_video_from_s3(video_clip_s3, video_clip, s3, retry=3):
    bucket_name = video_clip_s3.split('/')[2]
    file_name ='/'.join(video_clip_s3.split('/')[3:])
    for _ in range(retry):
        try:
            s3.download_file(bucket_name, file_name, video_clip)
            return
        except Exception as e:
            print(f'download {video_clip_s3} -> {video_clip} failed with {e}, sleep 5s')
            time.sleep(5)
            raise RuntimeError(f'download {video_clip_s3} -> {video_clip} failed')


def get_file(file_path, s3, prefix):

    with temp_file_context_manager('/dev/shm', prefix) as local_file_path:
        try:
            if 's3://' in file_path:
                download_video_from_s3(file_path, local_file_path, s3, 3)
                
                if prefix == '.mp4':
                    vr = decord.VideoReader(local_file_path)
                    return vr
                else:
                    mask_array = np.load(local_file_path)
                    return mask_array


        except Exception as e:
            print(f'Failed to read video: {file_path}')
            raise e


def collate_fn_with_diff_mask(batch):

    tar_video_key = torch.stack([item['tar_video_key'] for item in batch], dim=0)
    tar_video_key_mask = torch.stack([item['tar_video_key_mask'] for item in batch], dim=0)
    ref_video = torch.stack([item['ref_video'] for item in batch], dim=0)
    tar_video = torch.stack([item['tar_video'] for item in batch], dim=0)
    diff_mask = torch.stack([item['diff_mask'] for item in batch], dim=0)
    prompt = [item['prompt'] for item in batch]
    video_name = [item['video_name'] for item in batch]
    ref_img_path = [item['ref_img_path'] for item in batch]

    task_name = [item['task_name'] for item in batch]


    dict_data = {
        'tar_video_key': tar_video_key,
        'tar_video_key_mask': tar_video_key_mask,
        'ref_video': ref_video,
        'tar_video': tar_video,
        'diff_mask': diff_mask,
        'task_name': task_name,
        'prompt': prompt,
        'video_name': video_name,
        'ref_img_path': ref_img_path,
    }

    return dict_data



def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你的 csv 叫 data.csv
    csv_path = 'process_kiwidata_custom/remove/local_remove_mean_editscore_over6.csv'
    video_folder = "s3://hidream-user-zhangzhongwei/Dataset/OpenVE-3M/videos_folders" # 指向 local_add 所在的文件夹
    task_name = os.path.basename(csv_path).replace('.csv', '').replace("_mean_editscore_over6", "")
    
    dataset = VideoEditDatasetFromCSV(
        csv_path=csv_path, 
        height=480, width=832, max_num_frames=81, 
        base_video_folder=video_folder, # 指向 local_add 所在的文件夹
        task_name=task_name,
    )
    dataloader_train = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn_with_diff_mask)


    all_video_info_dict_list = []
    for i,data_iter in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):

        print(i)
        video_name = data_iter['video_name'][0]
        video_save_path = os.path.join('all_video_cache', f'{video_name}_tar.mp4')
        tar_video = (data_iter['tar_video'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255

        tar_video_list = [tar_video[i].astype(np.uint8) for i in range(tar_video.shape[0])]
        
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        save_video(tar_video_list, video_save_path, fps=16, quality=5)

        # ===========
        video_save_path_ref = os.path.join('all_video_cache', f'{video_name}_ref.mp4')
        ref_video = (data_iter['ref_video'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255

        ref_video_list = [ref_video[i].astype(np.uint8) for i in range(ref_video.shape[0])]

        os.makedirs(os.path.dirname(video_save_path_ref), exist_ok=True)
        save_video(ref_video_list, video_save_path_ref, fps=16, quality=5)

        # target_video_key
        video_save_path_target_key = os.path.join('all_video_cache', f'{video_name}_tar_key.mp4')
        tar_video_key = (data_iter['tar_video_key'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255
        tar_key_video_list = [tar_video_key[i].astype(np.uint8) for i in range(tar_video_key.shape[0])]
        os.makedirs(os.path.dirname(video_save_path_target_key), exist_ok=True)
        save_video(tar_key_video_list, video_save_path_target_key, fps=16, quality=5)

        # target_video_key_mask
        video_save_path_target_key_mask = os.path.join('all_video_cache', f'{video_name}_tar_key_mask.mp4')
        tar_video_key_mask = (data_iter['tar_video_key_mask'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255
        tar_video_key_mask_list = [tar_video_key_mask[i].astype(np.uint8) for i in range(tar_video_key_mask.shape[0])]
        os.makedirs(os.path.dirname(video_save_path_target_key_mask), exist_ok=True)
        save_video(tar_video_key_mask_list, video_save_path_target_key_mask, fps=16, quality=5)
        # ============

        prompt = data_iter['prompt'][0]
        txt_save_path = video_save_path.replace('.mp4', '.txt')
        with open(txt_save_path, 'w') as f:
            f.write(str(prompt))

        if i>50:
            break


