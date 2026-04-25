import torch
from torch.utils.data import DataLoader
import torch
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_dataset_unify import *
import numpy as np
import shutil
import imageio

def save_video(frames, save_path, fps, quality=5, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


rank = 0
world_size = 1


def get_kiwi_mixdata(rank=0, world_size=1, sample_prob_list=None):
    '''
    NOTE:
    - shuffle?
    - rank?
    - use_first_frame?

    '''
    class ReCo_remove_args:
        task_name="remove"
        json_path = 'kiwidata/remove/remove_data_configs_mean_editscore_over6.json' # YOUR JSON FOLDER...
        video_folder = 's3://hidream-user-zhangzhongwei/Dataset/ReCo_Data'

    class Openve_remove_args:
        csv_path = 'kiwidata/remove/local_remove_mean_editscore_over6.csv'
        video_folder = "s3://hidream-user-zhangzhongwei/Dataset/OpenVE-3M/videos_folders"
        task_name = os.path.basename(csv_path).replace('.csv', '').replace("_mean_editscore_over6", "")

    class Ditto_style_args:
        dataset_base_path="s3://hidream-dataset-ditto1m/Ditto-1M/videos/"
        dataset_metadata_path="kiwidata/style/ditto_csvs/global_style.csv"

    class SimpleArgs:
        dataset_folder_dict = {
            "Ditto-1M": "s3://hidream-dataset-ditto1m/Ditto-1M/videos",
            "openve3m": "s3://hidream-user-zhangzhongwei/Dataset/OpenVE-3M/videos_folders",
            "reco_data": "s3://hidream-user-zhangzhongwei/Dataset/ReCo_Data",
        }
        ref_img_folder="/mnt/zhongwei/Data/linyq/kiwi_edit_training_data/refvie_477k_stage3/ref_images"
        vid_ref_dataset_metadata_path = "kiwidata/kiwi-edit/refvie_dataset.csv"
        data_file_keys = "src_video,tgt_video,ref_image,prompt"
   
    # ---------------- Instance Datasets ------------------
 
    dataset_list = []
    # * Remove task
    # ------------------ ReCo remove --------------
    reco_args=ReCo_remove_args()
    reco_remove_dataaset = get_dataset_for_single_tasks(reco_args.json_path, reco_args.task_name, rank, world_size, \
                                            shuffle=True, base_video_folder=reco_args.video_folder, \
                                            read_video_from_local=False,
                                            )
    dataset_list.append(reco_remove_dataaset)

    # ------------------- OpenVE-3M local remove -----
    # openve_args=Openve_remove_args()
    # openve_remove_dataset = VideoEditDatasetFromCSV(
    #     csv_path=openve_args.csv_path, 
    #     rank=rank, world_size=world_size,
    #     height=480, width=832, max_num_frames=81, 
    #     base_video_folder=openve_args.video_folder, # 指向 local_add 所在的文件夹
    #     task_name=openve_args.task_name, shuffle=True,
    # )
    # dataset_list.append(openve_remove_dataset)

    # * Style
    # ------------------DITTO Style dataset ------------------
    ditto_args=Ditto_style_args()
    style_ditto = UnifiedDataset_ditto(
            base_path=ditto_args.dataset_base_path,
            metadata_path=ditto_args.dataset_metadata_path,
            filter_key_words=['style'],
            rank=rank, world_size=world_size,
            shuffle=True, user_first_frame=False,
        )
    dataset_list.append(style_ditto)

    # * kiwi-edit
    kiwi_args = SimpleArgs()
    kiwi_ref_dataset = UnifiedDataset_kiwidata(
        dataset_folder_dict=kiwi_args.dataset_folder_dict,
        ref_img_folder=kiwi_args.ref_img_folder,
        metadata_path=kiwi_args.vid_ref_dataset_metadata_path,
        data_file_keys=kiwi_args.data_file_keys.split(","),
        rank=rank, world_size=world_size,
        shuffle=True,
        user_first_frame=False,
    )
    dataset_list.append(kiwi_ref_dataset)

    dataset_mix = WebMixDatasetWithLength(dataset_list, sample_prob_list)
    dataloader_train = DataLoader(dataset_mix, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn_with_diff_mask)

    return dataloader_train


# # ************************ DEBUGS ***************************
# sample_prob_list = [0.0, 1.0, 1.0]
# dataloader_train = get_kiwi_mixdata(rank, world_size, sample_prob_list)

# from tqdm import tqdm
# for i, batch in tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
#     # print(f"\n--- 第 {i+1} 个样本 ---")
    
#     # if isinstance(batch, dict):
#     #     for key, value in batch.items():
#     #         if isinstance(value, torch.Tensor):
#     #             print(f"Key: {key:15} | Shape: {list(value.shape)} | Dtype: {value.dtype} | Scope: {value.max(), value.min()}")
#     #         else:
#     #             print(f"Key: {key:15} | Value Type: {type(value)}")
#     # else:
#     #     print(f"Batch 类型不是字典，是: {type(batch)}")

#     # ========== Start iter ==========
#     video_name = batch['video_name'][0]
#     video_save_path = os.path.join('all_cached/video_cache', f'{video_name}_key.mp4')
#     tar_video = (batch['tar_video_key'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255

#     tar_video_list = [tar_video[i].astype(np.uint8) for i in range(tar_video.shape[0])]
#     os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
#     save_video(tar_video_list, video_save_path, fps=16, quality=5)

#     video_save_path = os.path.join('all_cached/video_cache', f'{video_name}_key_mask.mp4')
#     diff_mask = batch['tar_video_key_mask'][0].permute(1,2,3,0).numpy()*255

#     diff_mask_list = [diff_mask[i].astype(np.uint8) for i in range(diff_mask.shape[0])]
#     os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
#     save_video(diff_mask_list, video_save_path, fps=16, quality=5)

#     # ========== Start iter ==========
#     video_name = batch['video_name'][0]
#     video_save_path = os.path.join('all_cached/video_cache', f'{video_name}.mp4')
#     tar_video = (batch['tar_video'][0].permute(1,2,3,0).numpy()/2 + 0.5)*255

#     tar_video_list = [tar_video[i].astype(np.uint8) for i in range(tar_video.shape[0])]
#     os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
#     save_video(tar_video_list, video_save_path, fps=16, quality=5)

#     # video_save_path = os.path.join('all_cached/video_cache', task_name, f'{video_name}_diff_mask.mp4')
#     # diff_mask = data_iter['diff_mask'][0].permute(1,2,3,0).numpy()*255

#     # diff_mask_list = [diff_mask[i].astype(np.uint8) for i in range(diff_mask.shape[0])]
#     # os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
#     # save_video(diff_mask_list, video_save_path, fps=16, quality=5)

#     # pil image save
#     pil_img_path = batch['ref_img_path'][0]
#     img_save_path = video_save_path.replace('.mp4', '.png')
#     if pil_img_path is not None:
#         shutil.copy(pil_img_path, img_save_path)


#     prompt = batch['prompt'][0]
#     txt_save_path = video_save_path.replace('.mp4', '.txt')
#     with open(txt_save_path, 'w') as f:
#         f.write(str(prompt))


#     if i>=200:
#         break

