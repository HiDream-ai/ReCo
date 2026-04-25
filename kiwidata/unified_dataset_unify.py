import torch, torchvision, imageio, os, json, pandas
import imageio.v3 as iio
from PIL import Image
import random
import boto3
import decord
import numpy as np
import torch.nn.functional as F
import contextlib
import uuid
import time
from torch.utils.data import IterableDataset
import torch.distributed as dist
from torch.utils.data import Dataset
import json
import pandas as pd



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


def get_dataset_for_single_tasks(json_path, task_name='replace', rank=0, world_size=1, shuffle=True, base_video_folder=None, read_video_from_local=True):

    # json_path = os.path.join(base_json_folder, task_name, f"{task_name}_data_configs.json")
    with open(json_path, "r", encoding="utf-8") as f:
        all_dict_list = json.load(f)

    if shuffle:
        import random
        rng = random.Random(time.time())
        rng.shuffle(all_dict_list)
    print(f'====== Process task name {task_name}, total {len(all_dict_list)} videos ======')

    sub_dict_list = all_dict_list[rank::world_size]
    # print(f'rank {rank} world_size {world_size} video num is: {len(sub_dict_list)}')

    video_dataset = ReCo_Dataset_train(all_data_list=sub_dict_list, base_video_folder=base_video_folder, read_video_from_local=read_video_from_local, task_name=task_name, user_first_frame=False)

    return video_dataset


def get_dataset_for_each_tasks(base_json_folder, task_name='replace', rank=0, world_size=1, shuffle=True, base_video_folder=None, read_video_from_local=True):

    json_path = os.path.join(base_json_folder, task_name, f"{task_name}_data_configs.json")
    with open(json_path, "r", encoding="utf-8") as f:
        all_dict_list = json.load(f)

    if shuffle:
        import random
        rng = random.Random(2026)
        rng.shuffle(all_dict_list)
    print(f'====== Process task name {task_name}, total {len(all_dict_list)} videos ======')

    sub_dict_list = all_dict_list[rank::world_size]
    print(f'rank {rank} world_size {world_size} video num is: {len(sub_dict_list)}')

    video_dataset = ReCo_Dataset_train(all_data_list=sub_dict_list, base_video_folder=base_video_folder, read_video_from_local=read_video_from_local, task_name=task_name)

    return video_dataset



class ReCo_Dataset_train(Dataset):
    def __init__(
        self,
        all_data_list: list = None,
        height: int = 480,
        width: int = 832,
        max_num_frames: int = 81,
        rank: int = 0,
        world_size: int = 1,
        base_video_folder: str = '',
        read_video_from_local = False,
        task_name = 'replace',
        user_first_frame=False,

    ) -> None:
        super().__init__()
        self.all_data_list = all_data_list
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.rank = rank
        self.world_size = world_size
        self.instance_metas = self.all_data_list[self.rank::self.world_size]
        self.s3 = boto3.client('s3')
        self.base_video_folder = base_video_folder
        self.read_video_from_local = read_video_from_local
        self.task_name = task_name
        self.user_first_frame = user_first_frame


    def __len__(self):
        return len(self.instance_metas)

    def read_video(self, video_path):

        vr = get_file(video_path, self.s3, prefix='.mp4')

        return vr

    def calculate_frame_indices(self, total_frames: int, video_fps: float, num_samples: int = 0) -> np.ndarray:

        beg = 0  
        end = total_frames

        frame_interval = video_fps / 16
        indices = np.arange(beg, end, frame_interval).astype(int)
        if len(indices) < num_samples:
            repeated_indices = np.array([indices[-1]] * (num_samples - len(indices)))
            indices = np.concatenate([indices, repeated_indices])
        
        if len(indices) > num_samples:
            frame_start = 0  
            frame_end = frame_start + num_samples
            indices = indices[frame_start:frame_end]       

        indices = np.clip(indices, 0, total_frames - 1)
        return indices


    def __getitem__(self, index):

        item = self.instance_metas[index]
        src_video_path = os.path.join(self.base_video_folder,item['src_video'])
        tar_video_path = os.path.join(self.base_video_folder, item['tar_video'])

        video_tar_name = os.path.basename(item['tar_video']).replace('.mp4','')
        video_src_name = os.path.basename(item['src_video']).replace('.mp4','')
        prompt = item['instruction_final_refine'].strip()
        task_name = self.task_name

        # -------------- STEP.1 read src video and process src video
        if not self.read_video_from_local:
            # load from aoss
            vr_src = self.read_video(src_video_path)
            vr_tar = self.read_video(tar_video_path)
        else:
            # load from local
            vr_src = decord.VideoReader(src_video_path)
            vr_tar = decord.VideoReader(tar_video_path)

        src_video = torch.from_numpy(vr_src.get_batch(list(range(0, len(vr_src)))).asnumpy())
        src_video = (src_video/255) *2 -1

        tar_video = torch.from_numpy(vr_tar.get_batch(list(range(0, len(vr_tar)))).asnumpy())
        tar_video = (tar_video/255) *2 -1

        tar_concated_video = torch.concat([src_video, tar_video], dim=2)

        # 3. return all data_dict
        f,h,w,c = tar_video.shape
        tar_video_key_mask = torch.zeros_like(tar_concated_video)
        # if self.user_first_frame and torch.rand(1).item()>0.95:
        #     tar_video_key_mask[1:,:,w:] = 1
        # else:
        tar_video_key_mask[:,:,w:] = 1

        tar_video_key = tar_concated_video * (1-tar_video_key_mask)
        ref_img_path = None

        video_save_name = f"task_{task_name}_dataset_reco_{video_tar_name}"

        return {
            'tar_video_key': tar_video_key.permute(3,0,1,2),                    # video input, [c,f,h,w], tensor [-1, 1]
            'tar_video_key_mask': tar_video_key_mask.permute(3,0,1,2),          # mask for tar_video_key, [c,f,h,w], 0-1
            'ref_video': torch.zeros_like(tar_video_key).permute(3,0,1,2),      # depth ect. input, [c,f,h,w], tensor [-1, 1]
            'tar_video': tar_concated_video.permute(3,0,1,2),                   # tar_video, [c,f,h,w], tensor [-1, 1]
            'diff_mask': torch.zeros_like(tar_video_key).permute(3,0,1,2),      # edit area mask, [c,f,h,w], tensor [-1, 1]
            'task_name': task_name,
            'prompt': prompt, 
            'video_name': video_save_name, 
            'ref_img_path': ref_img_path,
        }
    


class VideoEditDatasetFromCSV(Dataset):
    def __init__(
        self,
        csv_path: str,
        height: int = 480,
        width: int = 832,
        max_num_frames: int = 81,
        rank: int = 0,
        world_size: int = 1,
        base_video_folder: str = '',
        read_video_from_local = False,
        task_name = 'openve3m_local_add',
        user_first_frame = False,
        mask_config_json_path = None,
        shuffle = True,
    ) -> None:
        super().__init__()
        
        full_df = pd.read_csv(csv_path, sep=';')

        all_data = full_df.to_dict('records')
        print(f"Total {len(all_data)} videos for Openve Remove task")
        if shuffle:
            random.shuffle(all_data)
        self.instance_metas = all_data[rank::world_size]
        
        self.height = height
        self.width = width
        self.max_num_frames = max_num_frames
        self.base_video_folder = base_video_folder
        self.read_video_from_local = read_video_from_local
        self.task_name = task_name
        self.user_first_frame = user_first_frame
        self.s3 = boto3.client('s3')

    def __len__(self):
        return len(self.instance_metas)

    @staticmethod
    def resize_crop(video: torch.Tensor, oh: int, ow: int):
        """
        Resize, center crop and normalize for decord loaded video (torch.Tensor type)

        Parameters:
          video - video to process (torch.Tensor): Tensor from `reader.get_batch(frame_ids)`, in shape of (T, H, W, C)
          oh - target height (int)
          ow - target width (int)

        Returns:
            The processed video (torch.Tensor): Normalized tensor range [-1, 1], in shape of (C, T, H, W)

        Raises:
        """
        # permute ([t, h, w, c] -> [t, c, h, w])
        video = video.permute(0, 3, 1, 2)

        # resize and crop
        ih, iw = video.shape[2:]
        if ih != oh or iw != ow:
            # resize
            scale = max(ow / iw, oh / ih)
            video = F.interpolate(
                video,
                size=(round(scale * ih), round(scale * iw)),
                mode='bicubic',
                antialias=True
            )
            assert video.size(3) >= ow and video.size(2) >= oh

            # center crop
            x1 = (video.size(3) - ow) // 2
            y1 = (video.size(2) - oh) // 2
            video = video[:, :, y1:y1 + oh, x1:x1 + ow]

        # permute ([t, c, h, w] -> [c, t, h, w]) and normalize
        video = video.transpose(0, 1).float().div_(127.5).sub_(1.)
        return video


    def read_video(self, video_path):

        vr = get_file(video_path, self.s3, prefix='.mp4')

        return vr
    
    def __getitem__(self, index):
        item = self.instance_metas[index]

        tar_video_path = os.path.join(self.base_video_folder, item['video'])
        src_video_path = os.path.join(self.base_video_folder, item['original_video'])
        prompt = item['prompt'].strip()
        
        try:
            # Read videos
            vr_src = self.read_video(src_video_path)
            rounded_indices = np.round(np.linspace(0, len(vr_src) - 1, num=self.max_num_frames)).astype(int)
            src_video = torch.from_numpy(vr_src.get_batch(rounded_indices).asnumpy())

            vr_tar = self.read_video(tar_video_path)
            tar_video = torch.from_numpy(vr_tar.get_batch(rounded_indices).asnumpy())
            assert src_video.shape[0] == tar_video.shape[0], f"Frame count mismatch: {src_video.shape[0]} vs {tar_video.shape[0]}"

            # 2. resize and crop videos
            video_src_resized = self.resize_crop(src_video, self.height, self.width).permute(1,2,3,0)  # (T, H, W, C). [-1, 1]
            video_tar_resized = self.resize_crop(tar_video, self.height, self.width).permute(1,2,3,0)

            f,h,w,c = video_src_resized.shape
            tar_concated_video = torch.concat([video_src_resized, video_tar_resized], dim=2)

            # 3. return all data_dict
            tar_video_key_mask = torch.zeros_like(tar_concated_video)
            # if self.user_first_frame and torch.rand(1).item()>0.95:
            #     tar_video_key_mask[1:,:,w:] = 1
            # else:
            tar_video_key_mask[:,:,w:] = 1
            
            tar_video_key = tar_concated_video * (1 - tar_video_key_mask)
            video_tar_name = os.path.basename(item['video']).replace('.mp4','')
            video_save_name = f"task_{self.task_name}_{video_tar_name}"

            data = {
                'tar_video_key': tar_video_key.permute(3, 0, 1, 2),      # [C, F, H, W*2]
                'tar_video_key_mask': tar_video_key_mask.permute(3, 0, 1, 2), # [C, F, H, W*2]
                'ref_video': torch.zeros_like(tar_video_key).permute(3, 0, 1, 2),
                'tar_video': tar_concated_video.permute(3, 0, 1, 2),     # [C, F, H, W*2]
                'diff_mask': torch.zeros_like(tar_video_key).permute(3, 0, 1, 2),              # [C, F, H, W]
                'task_name': self.task_name,
                'prompt': prompt, 
                'video_name': video_save_name, 
                'ref_img_path': None,
            }

            return data
        except:
            print(f"Error loading data {index} {src_video_path} {tar_video_path}")
            index = random.randint(0, len(self.instance_metas) - 1) % len(self.instance_metas)
            return self.__getitem__(index)



class UnifiedDataset_ditto(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        filter_key_words=None,
        num_frames=81,
        height=480,
        width=832,
        rank: int = 0,
        world_size: int = 1,
        shuffle=True,
        user_first_frame=False,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.cached_data_operator = LoadTorchPickle()
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.data = self.load_metadata(metadata_path, filter_key_words)
        if shuffle:
            random.shuffle(self.data)
        print('============total number of style_ditto is: ', len(self.data))
        self.data = self.data[rank::world_size]
        
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.s3 = boto3.client('s3')
        self.user_first_frame=user_first_frame


    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path, filter_key_words=None):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            if filter_key_words is not None:
                for key_word in filter_key_words:
                    metadata = metadata[metadata['prompt'].str.contains(key_word, case=False, na=False)]

            data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        return data

    @staticmethod
    def resize_crop(video: torch.Tensor, oh: int, ow: int):
        """
        Resize, center crop and normalize for decord loaded video (torch.Tensor type)

        Parameters:
          video - video to process (torch.Tensor): Tensor from `reader.get_batch(frame_ids)`, in shape of (T, H, W, C)
          oh - target height (int)
          ow - target width (int)

        Returns:
            The processed video (torch.Tensor): Normalized tensor range [-1, 1], in shape of (C, T, H, W)

        Raises:
        """
        # permute ([t, h, w, c] -> [t, c, h, w])
        video = video.permute(0, 3, 1, 2)

        # resize and crop
        ih, iw = video.shape[2:]
        if ih != oh or iw != ow:
            # resize
            scale = max(ow / iw, oh / ih)
            video = F.interpolate(
                video,
                size=(round(scale * ih), round(scale * iw)),
                mode='bicubic',
                antialias=True
            )
            assert video.size(3) >= ow and video.size(2) >= oh

            # center crop
            x1 = (video.size(3) - ow) // 2
            y1 = (video.size(2) - oh) // 2
            video = video[:, :, y1:y1 + oh, x1:x1 + ow]

        # permute ([t, c, h, w] -> [c, t, h, w]) and normalize
        video = video.transpose(0, 1).float().div_(127.5).sub_(1.)
        return video

    def read_video(self, video_path):

        vr = get_file(video_path, self.s3, prefix='.mp4')

        return vr

    def __getitem__(self, data_id):
        
        data = self.data[data_id % len(self.data)].copy()
        video_src_path, video_tar_path, instruct_prompt = data['vace_video'], data['video'], data['prompt']

        video_name = os.path.basename(video_src_path).replace('.mp4', '')
        task_name = 'style'

        # * ------- 1. read decord and sampling video and masks
        if self.base_path is not None and self.base_path not in video_src_path:
            video_src_path = os.path.join(self.base_path, video_src_path)
            video_tar_path = os.path.join(self.base_path, video_tar_path)

        vr_src = self.read_video(video_src_path)
        rounded_indices = np.round(np.linspace(0, len(vr_src) - 1, num=self.num_frames)).astype(int)
        video_src = torch.from_numpy(vr_src.get_batch(rounded_indices).asnumpy())

        vr_tar = self.read_video(video_tar_path)
        video_tar = torch.from_numpy(vr_tar.get_batch(rounded_indices).asnumpy())

        # 2. resize and crop videos
        video_src_resized = self.resize_crop(video_src, self.height, self.width).permute(1,2,3,0)  # (T, H, W, C). [-1, 1]
        video_tar_resized = self.resize_crop(video_tar, self.height, self.width).permute(1,2,3,0)

        f,h,w,c = video_src_resized.shape
        tar_video = torch.concat([video_src_resized, video_tar_resized], dim=2)

        # 3. return all data_dict
        tar_video_key_mask = torch.zeros_like(tar_video)
        # if self.user_first_frame and torch.rand(1).item()>0.95:
        #     tar_video_key_mask[1:,:,w:] = 1
        # else:
        tar_video_key_mask[:,:,w:] = 1

        tar_video_key = tar_video * (1-tar_video_key_mask)
        ref_img_path = None
        video_save_name = f'task_{task_name}_dataset_DITTO_{video_name}'

        data = {
                'tar_video_key': tar_video_key.permute(3,0,1,2),                    # video input, [c,f,h,w], tensor [-1, 1]
                'tar_video_key_mask': tar_video_key_mask.permute(3,0,1,2),          # mask for tar_video_key, [c,f,h,w], 0-1
                'ref_video': torch.zeros_like(tar_video_key).permute(3,0,1,2),      # depth input, [c,f,h,w], tensor [-1, 1]
                'tar_video': tar_video.permute(3,0,1,2),                            # tar_video, [c,f,h,w], tensor [-1, 1]
                'diff_mask': torch.ones_like(video_src_resized).permute(3,0,1,2),
                'task_name': task_name,
                'prompt': instruct_prompt, 
                'video_name': video_save_name, 
                'ref_img_path': ref_img_path,
                }

        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True





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

DEBUG = False
# DEBUG = True

class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image

class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        if DEBUG: print(self.height, self.width, "height: ", height, "width: ", width)
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")

class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)

class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio



class WebMixDatasetWithLength(IterableDataset):
    
    def __init__(self, dataset_list, sample_prob_list, total_length=None):
        
        """
        A mixed iterable dataset combining multiple datasets
        
        Args:
                dataset_list: List of datasets, each element should be an iterable dataset
                sample_prob_list: Cumulative probability list, the last element should be 1.0
                                For example: [0.5, 0.8, 1.0] represents sampling ratios of 50%, 30%, 20%
        """
        super(WebMixDatasetWithLength, self).__init__()
        self.dataset_list = dataset_list
        self.sample_prob = sample_prob_list
        
        assert len(dataset_list) == len(sample_prob_list), "Number of datasets and probabilities must be equal"
        assert abs(sample_prob_list[-1] - 1.0) < 1e-6, "The last cumulative probability must be 1.0"

        self.dataset_iters = [iter(dataset) for dataset in dataset_list]
        if total_length is None:
            # Try to calculate total length
            self.total_length = 0
            for dataset, sample_prob in zip(dataset_list, sample_prob_list):
                if hasattr(dataset, '__len__'):
                    if sample_prob != 0.0:
                        self.total_length += len(dataset)
                    else:
                        self.total_length += 0
                else:
                    # For WebDataset and similar, may need manual specification
                    self.total_length = None
                    break
        else:
            self.total_length = total_length
    
    def __len__(self):
        return self.total_length

    def __iter__(self):
        """Finite iteration version"""

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        generator = torch.Generator()
        generator.manual_seed(2025 + rank + int(time.time()))

        # =============== START LOOP =================
        count = 0
        max_count = len(self) if self.total_length else float('inf')
        while count < max_count:

            # ------------ 1. Select dataset index based on probability
            rand_num = torch.rand(1, generator=generator).item()
            # print(f'rank: {rank}, random_num: {rand_num}')
            for i, prob in enumerate(self.sample_prob):
                if rand_num <= prob:
                    dataset_idx = i
                    break

            # ------------ 2. Get sample from the selected dataset
            try:
                sample = next(self.dataset_iters[dataset_idx])
                yield sample
                count += 1
            except StopIteration:
                print(f"🔄 Warning: Dataset at index {dataset_idx} is exhausted. Resetting it.")
                self.dataset_iters[dataset_idx] = iter(self.dataset_list[dataset_idx])


class UnifiedDataset_kiwidata(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_folder_dict=None, 
        ref_img_folder=None,
        metadata_path=None,
        data_file_keys=tuple(),
        num_frames=81, height=480, width=832,
        rank=0, world_size=1,
        user_first_frame=False,
        shuffle=False,
        remove_key_words=None,
    ):
        self.dataset_folder_dict = dataset_folder_dict
        self.ref_img_folder = ref_img_folder
        self.metadata_path = metadata_path
        self.data_file_keys = data_file_keys
        self.cached_data = []
        self.remove_key_words = remove_key_words
        data = self.load_metadata(metadata_path)
        if shuffle:
            random.shuffle(data)
        print('============total number of kiwi_edit data is: ', len(data))
        self.data = data[rank::world_size]

        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.s3 = boto3.client('s3')
        self.user_first_frame = user_first_frame


    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

            # metadata = pandas.read_csv(metadata_path)

            # # filter out diff folder_name

            # # 过滤：src_video 路径中包含 "Ditto-1M/local" 的行删掉
            # if "src_video" in metadata.columns and self.remove_key_words is not None:
            #     before = len(metadata)
            #     mask = ~metadata["src_video"].astype(str).str.contains("Ditto-1M/local", na=False)
            #     metadata = metadata[mask].reset_index(drop=True)
            #     after = len(metadata)
            #     if before != after:
            #         print(f"Filtered out {before - after} rows with src_video containing 'Ditto-1M/local'.")
                
            #     # return dict list
            #     data = metadata.to_dict(orient="records")
            # else:
            #     # metadata = pandas.read_csv(metadata_path)
            #     data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

        return data


    @staticmethod
    def resize_crop(video: torch.Tensor, oh: int, ow: int):
        """
        Resize, center crop and normalize for decord loaded video (torch.Tensor type)

        Parameters:
          video - video to process (torch.Tensor): Tensor from `reader.get_batch(frame_ids)`, in shape of (T, H, W, C)
          oh - target height (int)
          ow - target width (int)

        Returns:
            The processed video (torch.Tensor): Normalized tensor range [-1, 1], in shape of (C, T, H, W)

        Raises:
        """
        # permute ([t, h, w, c] -> [t, c, h, w])
        video = video.permute(0, 3, 1, 2)

        # resize and crop
        ih, iw = video.shape[2:]
        if ih != oh or iw != ow:
            # resize
            scale = max(ow / iw, oh / ih)
            video = F.interpolate(
                video,
                size=(round(scale * ih), round(scale * iw)),
                mode='bicubic',
                antialias=True
            )
            assert video.size(3) >= ow and video.size(2) >= oh

            # center crop
            x1 = (video.size(3) - ow) // 2
            y1 = (video.size(2) - oh) // 2
            video = video[:, :, y1:y1 + oh, x1:x1 + ow]

        # permute ([t, c, h, w] -> [c, t, h, w]) and normalize
        video = video.transpose(0, 1).float().div_(127.5).sub_(1.)
        return video

    def read_video(self, video_path):

        vr = get_file(video_path, self.s3, prefix='.mp4')

        return vr
    
    def get_base_folder_fix_path(self, video_src_path: str, video_tar_path: str):

        for key, value in self.dataset_folder_dict.items():
            if key in video_src_path:
                base_folder = value

                if key == "Ditto-1M":
                    src_rel = video_src_path.replace("video/Ditto-1M/", "")
                    tar_rel = video_tar_path.replace("video/Ditto-1M/", "")
                elif key == "openve3m":
                    src_rel = video_src_path.replace("video/openve3m/", "")
                    tar_rel = video_tar_path.replace("video/openve3m/", "")
                elif key == "reco_data":
                    src_rel = video_src_path.replace("video/reco_data/source/", "")
                    tar_rel = video_tar_path.replace("video/reco_data/target/", "")
                else:
                    NotImplementedError

                src_path = os.path.join(base_folder, src_rel)
                tar_path = os.path.join(base_folder, tar_rel)

                return src_path, tar_path

        return None


    def __getitem__(self, data_id):
        
        try: 
            data = self.data[data_id % len(self.data)].copy()
            video_src_path, video_tar_path, instruct_prompt = data['src_video'], data['tgt_video'], data['prompt']
            ref_img_path = data['ref_image']

            # Read video name and fix video path
            video_name = os.path.basename(ref_img_path).replace('.jpg', '')
            video_src_path, video_tar_path = self.get_base_folder_fix_path(video_src_path, video_tar_path)

            # Read videos
            vr_src = self.read_video(video_src_path)
            rounded_indices = np.round(np.linspace(0, len(vr_src) - 1, num=self.num_frames)).astype(int)
            video_src = torch.from_numpy(vr_src.get_batch(rounded_indices).asnumpy())

            vr_tar = self.read_video(video_tar_path)
            video_tar = torch.from_numpy(vr_tar.get_batch(rounded_indices).asnumpy())

            # 2. resize and crop videos
            video_src_resized = self.resize_crop(video_src, self.height, self.width).permute(1,2,3,0)  # (T, H, W, C). [-1, 1]
            video_tar_resized = self.resize_crop(video_tar, self.height, self.width).permute(1,2,3,0)

            f,h,w,c = video_src_resized.shape
            tar_video = torch.concat([video_src_resized, video_tar_resized], dim=2)

            # 3. return all data_dict
            tar_video_key_mask = torch.zeros_like(tar_video)
            # if self.user_first_frame and torch.rand(1).item()>0.95:
            #     tar_video_key_mask[1:,:,w:] = 1
            # else:
            tar_video_key_mask[:,:,w:] = 1

            tar_video_key = tar_video * (1-tar_video_key_mask)
            video_save_name = f'{video_name}'

            data = {
                    'tar_video_key': tar_video_key.permute(3,0,1,2),                    # video input, [c,f,h,w], tensor [-1, 1]
                    'tar_video_key_mask': tar_video_key_mask.permute(3,0,1,2),          # mask for tar_video_key, [c,f,h,w], 0-1
                    'ref_video': torch.zeros_like(tar_video_key).permute(3,0,1,2),      # depth input, [c,f,h,w], tensor [-1, 1]
                    'tar_video': tar_video.permute(3,0,1,2),                            # tar_video, [c,f,h,w], tensor [-1, 1]
                    'diff_mask': torch.ones_like(video_src_resized).permute(3,0,1,2),
                    'task_name': video_name,
                    'prompt': instruct_prompt, 
                    'video_name': video_save_name, 
                    'ref_img_path': ref_img_path,
                    }

            return data
        except:
            print(f"Error loading data {data_id} {video_src_path} {video_tar_path}")
            data_id = random.randint(0, len(self.data) - 1) % len(self.data)
            return self.__getitem__(data_id)

    def __len__(self):
        return len(self.data)
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
    def check_paired_size(self, data1, data2):
        err_message = ""
        if data1[0].size[0] != data2[0].size[0]:
            err_message += f'mismatch width size {data1[0].size[0]} {data2[0].size[0]}'
        if data1[0].size[1] != data2[0].size[1]:
            err_message += f'mismatch height size {data1[0].size[1]} {data2[0].size[1]}'
        if len(data1) != len(data2):
            err_message += f'mismatch frame length {len(data1)} {len(data2)}'
        return err_message


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        max_retry = 30
        retry_count = 0
        while retry_count < max_retry:
            try:
                if self.load_from_cache:
                    data = self.cached_data[data_id % len(self.cached_data)]
                    data = self.cached_data_operator(data)
                else:
                    data = self.data[data_id % len(self.data)].copy()
                    src_key = data['src_video']
                    tgt_key = data['tgt_video']
                    for key in self.data_file_keys:
                        if key in data:
                            if key in self.special_operator_map:
                                data[key] = self.special_operator_map[key](data[key])
                            elif key in self.data_file_keys:
                                if isinstance(data[key], list):
                                    data[key] = [self.main_data_operator(item)[0] for item in data[key]]
                                else:
                                    data[key] = self.main_data_operator(data[key])
                err_message = self.check_paired_size(data['src_video'], data['tgt_video'])
                if err_message:
                    raise ValueError(err_message)
                return data
            except Exception as e:
                print(f"Error {retry_count}/{max_retry} loading data {data_id} {src_key} {tgt_key}: {e}")
                retry_count += 1
                data_id = random.randint(0, len(self.data) - 1) % len(self.data)
                continue
        raise ValueError(f"Failed to load data {data_id} after {max_retry} retries.")

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
    
    def check_paired_size(self, data1, data2):
        err_message = ""
        if data1[0].size[0] != data2[0].size[0]:
            err_message += f'mismatch width size {data1[0].size[0]} {data2[0].size[0]}'
        if data1[0].size[1] != data2[0].size[1]:
            err_message += f'mismatch height size {data1[0].size[1]} {data2[0].size[1]}'
        if len(data1) != len(data2):
            err_message += f'mismatch frame length {len(data1)} {len(data2)}'
        return err_message
