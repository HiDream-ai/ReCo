import torch
from torch.utils.data import DataLoader
from diffsynth.trainers.unified_dataset import UnifiedDataset


# 1. 直接用你提供的参数进行设置
class SimpleArgs:
    dataset_base_path = "./demo_data"
    vid_ref_dataset_metadata_path = "./demo_data/video_ref_demo_training_set.csv"
    num_frames = 81
    dataset_repeat = 1
    data_file_keys = "src_video,tgt_video,ref_image"
    max_pixels = 921600
    height = 720
    width = 1280

args = SimpleArgs()


# 2. 实例化数据集 (确保 UnifiedDataset 已导入)
vid_ref_dataset = UnifiedDataset(
    base_path=args.dataset_base_path,
    metadata_path=args.vid_ref_dataset_metadata_path,
    repeat=args.dataset_repeat,
    data_file_keys=args.data_file_keys.split(","),
    main_data_operator=UnifiedDataset.default_video_operator(
        base_path=args.dataset_base_path,
        max_pixels=args.max_pixels,
        height=args.height,
        width=args.width,
        height_division_factor=32,
        width_division_factor=32,
        num_frames=args.num_frames,
        time_division_factor=4,
        time_division_remainder=1,
    ),
)

# 3. 创建 DataLoader
# 注意：num_workers=0 在调试时最稳，不容易报错
vid_ref_dataloader = DataLoader(vid_ref_dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=0)

# 4. 简单迭代看一眼
print("开始检查数据集...")
for i, batch in enumerate(vid_ref_dataloader):
    print(f"\n--- 第 {i+1} 个样本 ---")
    
    if isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"Key: {key:15} | Shape: {list(value.shape)} | Dtype: {value.dtype}")
            else:
                print(f"Key: {key:15} | Value Type: {type(value)}")
    else:
        print(f"Batch 类型不是字典，是: {type(batch)}")

    # 只看前 3 个样本就停
    if i >= 2:
        break

print("\n检查完毕！")