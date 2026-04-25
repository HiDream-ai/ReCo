import os
# 限制 BLAS / OpenMP 并行度
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import torch, os, argparse
import lightning as pl
from diffsynth import WanVideoPipeline, ModelManager_custom, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np
import torch.nn.functional as F
from diffsynth import save_video
from torch.utils.data import DataLoader
import torch.distributed as dist
import random
from sklearn.cluster import MiniBatchKMeans
import einops
from threadpoolctl import threadpool_limits
import math
import cv2
from reco_data_test_mix_data import ReCo_Dataset_train, collate_fn_with_diff_mask, WebMixDatasetWithLength
from kiwidata.test_dataset_mixdata import get_kiwi_mixdata


def set_seed(seed, rank):
    adjusted_seed = seed + rank
    random.seed(adjusted_seed)
    np.random.seed(adjusted_seed)
    torch.manual_seed(adjusted_seed)
    torch.cuda.manual_seed_all(adjusted_seed)


import json
import time
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



class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None, 
        log_video_steps=500,
        train_from_sratch=False,
        add_mask_loss=False,
        # imgedit_base_folder=None,
        # task_name_list=None,
        single_task=None,
        height=None,
        width=None,
        num_frames=None,
        train_batch_size=1,
        use_contrast_loss=False,
        use_mse_loss=False,
        use_attnscore_loss=False,
        debug_ornot=False,
        dataloader_num_workers=0,
    ):
        super().__init__()

        self.train_architecture = train_architecture
        self.log_video_steps = log_video_steps
        self.add_mask_loss = add_mask_loss
    
        self.single_task = single_task
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.train_batch_size = train_batch_size
        self.use_contrast_loss = use_contrast_loss
        self.use_mse_loss = use_mse_loss
        self.use_attnscore_loss = use_attnscore_loss
        self.debug = debug_ornot
        self.dataloader_num_workers = dataloader_num_workers

        # Prepare: model_structure, model_detector, model_name, model_path
        # ! Change..
        model_manager = ModelManager_custom(torch_dtype=torch.bfloat16, device="cpu", train=train_from_sratch)
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models(dit_path)             # NOTE: change from ModelManager for change model structure..   
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Freeze parameters
        self.pipe.requires_grad_(False)

        # set requires_grad_ params for optimizer
        if train_architecture == "lora":                    # NOTE: only add lora for VACE controlnet
            self.add_lora_to_model(
                self.pipe.vace,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,                      # alpha/r * BA
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        elif train_architecture == "all_lora":
            self.add_lora_to_model(
                self.pipe.vace,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,                      # alpha/r * BA
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,                      # alpha/r * BA
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        elif train_architecture == "vace":
            self.pipe.vace.requires_grad_(True)
        elif train_architecture == "full":
            # Merge Lora params                   # NOTE: PEFT can't move model params totally
            if pretrained_lora_path is not None:
                state_dict = load_state_dict(pretrained_lora_path)

                # load vace lora params
                state_dict_lora = {k: v for k, v in state_dict.items() if "vace" in k}
                model_manager.load_lora(state_dict=state_dict_lora)

                # load dit lora params
                state_dict_dit = {k: v for k, v in state_dict.items() if "vace" not in k}
                model_manager.load_lora(state_dict=state_dict_dit)

            self.pipe.vace.requires_grad_(True)
            self.pipe.denoising_model().requires_grad_(True)
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    def train_dataloader(self):
        # step 1
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        # sample_prob_list = [0.15, 0.25, 0.5, 1.0] if not self.debug else [0, 0, 0, 1.0]
        sample_prob_list = [0.2, 0.5, 1.0] if not self.debug else [0, 0, 1.0]
        
        dataloader_train = get_kiwi_mixdata(rank, world_size, sample_prob_list=sample_prob_list)
        return dataloader_train


    def get_processed_ref_img(self, ref_img_path, height, width, device):

        random_value = torch.rand(1).item()
        if ref_img_path is not None and random_value<0.75:   # NOTE: 70% chance to use ref image, 30% chance to not use ref image, for better generalization..can adjust the ratio.
            ref_img_pil = Image.open(ref_img_path).convert("RGB")

            # ---- 随机增强：水平/垂直翻转 + 小角度旋转 ----
            # if random.random() < 0.1:
            #     ref_img_pil = ImageOps.mirror(ref_img_pil)      # 左右翻转
            # if random.random() < 0.1:
            #     ref_img_pil = ImageOps.flip(ref_img_pil)        # 上下翻转
            # if random.random() < 0.2:
            #     angle = random.uniform(-45, 45)                     # 可按需调范围
            #     ref_img_pil = ref_img_pil.rotate(
            #         angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)
            #     )
            # --------------------------------------------------

            ref_width, ref_height = ref_img_pil.size
            canvas_height, canvas_width = height, width

            if (ref_height, ref_width) != (canvas_height, canvas_width):
                scale = min(canvas_height / ref_height, canvas_width / ref_width)
                new_height = int(ref_height * scale)
                new_width = int(ref_width * scale)

                resized_pil = ref_img_pil.resize((new_width, new_height), Image.LANCZOS)
                white_canvas_pil = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

                top = (canvas_height - new_height) // 2
                left = (canvas_width - new_width) // 2
                white_canvas_pil.paste(resized_pil, (left, top))
                white_canvas_np = np.array(white_canvas_pil)

                ref_img_pil = Image.fromarray(np.concatenate([np.ones_like(white_canvas_np)*255, white_canvas_np], axis=1))
        else:
            ref_img_pil = None

        return ref_img_pil


    @torch.no_grad()
    def get_obj_latent_mask_from_video(self, mask_tensor, video_save_name=None, batch_idx=0, dropout_or=False):
        
        # torch.Size([1, 3, 81, 480, 832]), tensor [max, min] --> [1,-1], torch.bf16
        mask_latents = self.pipe.encode_video(mask_tensor.to(self.pipe.torch_dtype), tiled=False).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        
        # 3. cluster latent_mask to get binary latent_mask
        if dropout_or:
            p1, p2 = 2,2
        else:
            p1, p2 = 1,1        # ori resolution..

        mask_latents = mask_latents.permute(0, 2, 3, 4, 1)
        mask_latents = einops.rearrange(mask_latents, 'B F (H p1) (W p2) C -> B F H W (p1 p2 C)', p1=p1, p2=p2)
        bs, frame, h, w, channel  = mask_latents.shape
        latent_flat = mask_latents.view(-1, channel).to(torch.float32).cpu().numpy()
        
        # ---------ori version, large cost, change minibatch version..
        # n_clusters = 2
        # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # kmeans.fit(latent_flat)
        # cluster_labels = kmeans.labels_
        
        # low cost
        n_clusters = 2
        with threadpool_limits(limits=1, user_api='blas'):  # 1 或 2/4
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                    batch_size=65536, max_iter=100)
            kmeans.fit(latent_flat)
            cluster_labels = kmeans.labels_

        # 计算每个簇在 latent_flat 上的均值
        cluster_means = [latent_flat[cluster_labels == i].mean() for i in range(n_clusters)]

        # 建立映射：均值大的簇映射为 1，均值小的簇映射为 0
        mapping = {np.argmax(cluster_means): 1, np.argmin(cluster_means): 0}
        cluster_labels = np.vectorize(mapping.get)(cluster_labels)
        
        latent_mask = torch.from_numpy(cluster_labels.reshape(bs, frame, h, w, 1)).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)

        # latent_mask = 1 - latent_mask
        obj_latent_mask = latent_mask
        base_output_path = f"{os.path.dirname(self.trainer.checkpoint_callback.dirpath)}/diff_mask_video/gt/steps_{batch_idx}_{video_save_name}"
        video_mask_save_path = os.path.join(base_output_path, 'new_latent_masks')

        # 4. visual latent mask
        if batch_idx%250==0:
            os.makedirs(video_mask_save_path, exist_ok=True)
            for i in range(latent_mask.shape[1]):
                per_frame = (latent_mask[0][i] * 255).to(torch.uint8).repeat(1,1,3).detach().cpu().numpy()
                per_frame_pil = Image.fromarray(per_frame)
                per_frame_pil.save(os.path.join(video_mask_save_path, f'{i:05d}.png'))
        
        # Done.  Return..latent_mask----torch.Size([1, f, 60, 90, 1]) --> [1,1,f*h*w,c]
        return obj_latent_mask


    def mask_separation_loss(self, diff_tensor, mask_tensor):
        """
        A: [B, C, F, H, W]  差值（可以是正负）
        M: [B, 1, F, H, W]  mask (0/1 或 [0,1])

        返回：标量 loss
        """
        eps = 1e-6

        # 可选：用幅度来衡量“差值大小”
        a_mag = diff_tensor.abs()                 # 也可换成 A.pow(2) 或 A.norm(dim=1)

        # 也可以先把通道聚合成单通道“强度”
        a_score = a_mag.mean(dim=1, keepdim=True)  # [B,1,F,H,W]
        fg = mask_tensor.float()                   # 前景权重
        bg = 1.0 - fg                    # 背景权重

        # 按权重求均值（对全维做加权平均）
        def wmean(input, weight):
            num = (input * weight).sum(dim=(1,2,3,4))                 # [B]
            den = weight.sum(dim=(1,2,3,4)).clamp_min(eps)        # [B]
            return num / den                                 # [B]

        mean_fg = wmean(a_score, fg)     # mean(A[M])
        mean_bg = wmean(a_score, bg)     # mean(A[1-M])

        # 基本版（最小化它，使 mean_fg > mean_bg）
        loss_vec = mean_bg - mean_fg

        return loss_vec.mean()

    @torch.no_grad()
    def save_norm_percentile_mask(self, x, video_name, timestep_id, diff_loss, p=0.85, normalize_by_c=True, batch_idx=0):
        """
        x: [B, C, F, H, W] （建议先 x = x.float() 再传入）
        p: 分位数；0.90 表示“取每个样本前 10% 的位置为 1”
        normalize_by_c: 是否用 ||x||2 / sqrt(C) 做尺度对齐
        batch_index: 保存第几个 batch（一般 0）
        """
        assert x.dim() == 5, f"expect [B,C,F,H,W], got {tuple(x.shape)}"
        b, c = x.shape[:2]

        # 1) score = ||x||2（跨通道聚合），可选除以 sqrt(C) 让不同 C 可比
        score = x.float().norm(dim=1, keepdim=True)                  # [B,1,F,H,W]
        if normalize_by_c:
            score = score / math.sqrt(c)

        # 2) 每个样本独立用分位数做阈值
        flat = score.view(b, -1)                                     # [B, 1*F*H*W]
        q = torch.quantile(flat, q=p, dim=1)                         # [B]
        tau = q.view(b, 1, 1, 1, 1)                                  # [B,1,1,1,1]
        mask_bool = (score > tau)                                    # [B,1,F,H,W], bool

        # 3) 准备保存用的 0/1 张量（去掉通道维）→ [B,F,H,W]
        diff_mask_tensor = mask_bool.to(torch.uint8).squeeze(1)      # 0/1

        # 4) 保存为 PNG（与你的路径和写法对齐）
        base_output_path = f"{os.path.dirname(self.trainer.checkpoint_callback.dirpath)}/diff_mask_video/pred/steps_{batch_idx}_timestep_{timestep_id.item()}_diff_loss_{diff_loss:.3f}_{video_name}"
        video_mask_save_path = os.path.join(base_output_path, 'new_latent_masks')
        os.makedirs(video_mask_save_path, exist_ok=True)

        # 只保存 batch_index 这一条视频
        # assert 0 <= batch_index < b
        for i in range(diff_mask_tensor.shape[1]):                   # 遍历帧 F
            # 取出 [H,W] -> 放大到 0/255，再堆成 3 通道
            # per_frame = (diff_mask_tensor[batch_index, i] * 255).detach().cpu().numpy()
            # per_frame_pil = Image.fromarray(per_frame).convert("L")  # 先存灰度更小
            # 如果你一定要 3 通道和原写法一致：改为下面三行
            per_frame = (diff_mask_tensor[0, i] * 255).to(torch.uint8).unsqueeze(-1).repeat(1,1,3).cpu().numpy()
            per_frame_pil = Image.fromarray(per_frame)
            # ↑ 这会生成 3 通道 PNG（体积更大）

            per_frame_pil.save(os.path.join(video_mask_save_path, f'{i:05d}.png'))

        return mask_bool, diff_mask_tensor  # [B,1,F,H,W], [B,F,H,W]

    def save_x1(self, pred_x1, video_name, timestep_id, diff_loss, batch_idx=0):

        pred_x1_pixel = self.pipe.decode_video(pred_x1, tiled=False).clamp_(-1.0, 1.0)
        pred_x1_pixel = (pred_x1_pixel+1)*0.5*255
        pred_x1_pixel = pred_x1_pixel[0].permute(1,2,3,0).to(torch.uint8)

        base_output_path = f"{os.path.dirname(self.trainer.checkpoint_callback.dirpath)}/diff_mask_video/pred/steps_{batch_idx}_timestep_{timestep_id.item()}_diff_loss_{diff_loss:.3f}_{video_name}.mp4"
        os.makedirs(os.path.dirname(base_output_path), exist_ok=True)

        torchvision.io.write_video(
            base_output_path,
            pred_x1_pixel,          # uint8, [T,H,W,C]
            fps=16,
            video_codec='libx264',
            options={'crf': '20'}  # 品质-体积权衡，可调 18~28
        )

        return pred_x1_pixel

    def get_hull_for_diff_mask(self, x: torch.Tensor, thresh=0.0, out_range='-1/1'):

        """
        x: (C, F, H, W)，值在 [-1,1]
        返回每个独立区域的凸包并集（每区域独立凸包）。
        out_range: '-1/1' 或 '0/1'
        """
        assert x.dim() == 4, "expect (C,F,H,W)"
        C, F, H, W = x.shape
        dev, dtype = x.device, x.dtype

        bin_np = (x.detach().to(torch.float32).cpu().numpy() > thresh)   # (C,F,H,W) bool
        out_np = np.zeros((C, F, H, W), dtype=np.uint8)     # 0/1

        for c in range(C):
            for f in range(F):
                m = (bin_np[c, f].astype(np.uint8) * 255)   # 0/255
                if m.any():
                    # 只取外部轮廓：每个连通区域一条轮廓（孔洞忽略）
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    dst = np.zeros((H, W), dtype=np.uint8)
                    for cnt in cnts:
                        if cnt.shape[0] >= 3:              # 至少三个点才能形成凸包
                            hull = cv2.convexHull(cnt)
                            if hull is not None and hull.shape[0] >= 3:
                                cv2.fillPoly(dst, [hull], 1)
                    out_np[c, f] = dst
                else:
                    out_np[c, f] = 0

        out = torch.from_numpy(out_np).to(dev)
        if out_range == '-1/1':
            out = out.to(dtype).mul(2).sub(1)              # 0/1 -> -1/1
        else:
            out = out.to(dtype)
        return out

    def dilate_mask_bfhwc(self, mask_bf: torch.Tensor, radius: int = 10) -> torch.Tensor:
        """
        支持 (F,H,W,C) 或 (B,F,H,W,C)。值域 -1/1 或 0/1。
        返回同形状张量，做 k=(2*radius+1) 的矩形膨胀（逐通道）。
        """
        assert mask_bf.ndim in (4, 5), "expect (F,H,W,C) or (B,F,H,W,C)"
        if mask_bf.ndim == 4:
            F_, H, W, C = mask_bf.shape
            x = mask_bf.view(1, F_, H, W, C)
        else:
            B, F_, H, W, C = mask_bf.shape
            x = mask_bf

        # 合并 B 和 F → (BF, H, W, C)
        x = x.reshape(-1, H, W, C)

        # NHWC → NCHW
        x = x.permute(0, 3, 1, 2)                     # (BF, C, H, W)
        bin01 = (x > 0).to(torch.float32)

        k = 2 * radius + 1
        y = F.max_pool2d(bin01, kernel_size=(1, k), stride=1, padding=(0, radius))
        y = F.max_pool2d(y,     kernel_size=(k, 1), stride=1, padding=(radius, 0))

        out = (y * 2 - 1).to(mask_bf.dtype).permute(0, 2, 3, 1)  # (BF,H,W,C)

        # 还原形状
        if mask_bf.ndim == 4:
            return out.view(F_, H, W, C)
        else:
            return out.view(B, F_, H, W, C)

    def training_step(self, batch, batch_idx):
        # Data
        with torch.no_grad():
            self.pipe.device = self.device
            prompt = batch["prompt"]           
            self.pipe.load_models_to_device(["text_encoder"])
            prompt_emb = self.pipe.encode_prompt(prompt)                # ! NOTE: CHANGE
            
            ref_img_pil = self.get_processed_ref_img(batch["ref_img_path"][0],self.height, self.width, device=self.pipe.device)    # bs = 1, NOTE: input ip_images
            latents, vace_kwargs = self.pipe.prepare_vace_kwargs_new(
                vace_video=batch["tar_video_key"], vace_video_ref=batch["ref_video"],       # dataset return video_name, ref_video, tar_video etc.
                vace_mask=batch["tar_video_key_mask"], tar_video=batch['tar_video'], ref_img_pil=ref_img_pil, inference=False, video_name=batch['video_name'][0]
            )
            # acc task name use diff loss
            task_name = batch['task_name'][0]
            video_save_name = batch['video_name'][0]

            if self.use_contrast_loss or self.use_mse_loss or self.use_attnscore_loss:
                if 'style' in task_name:
                    latent_mask = torch.ones((1,21,60,104,1), dtype=self.pipe.torch_dtype, device=self.pipe.device)
                else:
                    latent_mask = self.get_obj_latent_mask_from_video(batch["diff_mask"], video_save_name, batch_idx)
            else:
                latent_mask = None


        bs, c, f, h, w = latents.shape
        # Loss
        self.pipe.load_models_to_device(["dit", "vace"])
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (bs,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep_id)
        training_target = self.pipe.scheduler.training_target(latents, noise)   # vector..

        latents = latents.to(dtype=self.pipe.torch_dtype, device=self.device)
        noisy_latents = noisy_latents.to(dtype=self.pipe.torch_dtype, device=self.device)

        # Compute loss
        self.pipe.dit.train()
        self.pipe.vace.train()
        if self.use_attnscore_loss:
            try:
                noise_pred, attnscore_list = self.pipe.model_fn_wan_video_w_attnscore(
                    self.pipe.dit, vace=self.pipe.vace, 
                    x=noisy_latents, timestep=timestep,         # x->torch.Size([1, 16, 22, 60, 104], -5,5)
                    **prompt_emb, **extra_input, **vace_kwargs,
                    use_gradient_checkpointing=self.use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
                    latent_mask=latent_mask,
                )
            except Exception as e:
                print(f'task_name: {task_name}')
                print(f'input noise_latents shape: {noisy_latents.shape}')
                print(f'latent mask shape: {latent_mask.shape}')
                print(f'Error in model_fn_wan_video_w_attnscore: {e}')
                raise e
        else:
            noise_pred = self.pipe.model_fn_wan_video(
                self.pipe.dit, vace=self.pipe.vace, 
                x=noisy_latents, timestep=timestep,         # x->torch.Size([1, 16, 22, 60, 104], -5,5)
                **prompt_emb, **extra_input, **vace_kwargs,
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
            )

        # ============================== Define loss =======================
        if latent_mask is not None:
            latent_mask_new = torch.concat([latent_mask]*2, dim=3).permute(0,4,1,2,3)
        else:
            latent_mask_new = None
        
        loss_mse, diff_loss, loss_attnscore = 0.0, 0.0, 0.0

        # ============== 1. Define base MSE Loss
        if noise_pred.shape[2]!=21:
            loss = torch.nn.functional.mse_loss(noise_pred[:,:,1:].float(), training_target[:,:,1:].float())
            loss_base = loss.detach().cpu().item()
        else:
            loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())   # NOTE: equal loss
            loss_base = loss.detach().cpu().item()
        
        # ============== 2. Define strengthen MSE Loss for edit area
        if self.use_mse_loss and latent_mask_new is not None and 'style' not in task_name:                   # also use for style..
        # if self.use_mse_loss and latent_mask_new is not None:
            if noise_pred.shape[2]!=21:
                loss_mse = torch.nn.functional.mse_loss((noise_pred[:,:,1:]*latent_mask_new).float(), (training_target[:,:,1:]*latent_mask_new).float())
            else:
                loss_mse = torch.nn.functional.mse_loss((noise_pred*latent_mask_new).float(), (training_target*latent_mask_new).float())

            loss = loss + loss_mse 

        # loss = loss * self.pipe.scheduler.training_weight(timestep)           # No weight...
        lr = self.optimizers().param_groups[0]["lr"] 
        # Record log
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.log_dict(
                {f"loss": loss, "loss_base": loss_base, "loss_mse": loss_mse, "diff_loss": diff_loss, 'loss_attnscore': loss_attnscore, "lr": lr}, 
                prog_bar=True
                )
        
        del prompt_emb, latents, noisy_latents, training_target
        return loss


    @torch.no_grad()
    def run_val_func(self, batch, batch_idx):

        global_rank = self.trainer.global_rank
        self.pipe.eval()
        self.pipe.device = self.device

        video_names = batch["video_name"]
        # task_name = batch["task_name"]
        # dataset_type = batch["dataset_type"]
        b,c,f,h,w = batch["tar_video_key"].shape

        for i in range(b):

            # Depth video + Reference image -> Video
            ref_img_pil = self.get_processed_ref_img(batch["ref_img_path"][0],self.height, self.width, device=self.pipe.device)    # bs = 1, NOTE: input ip_images
            negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

            video = self.pipe(
                prompt=batch["prompt"][i],
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                height=h, width=w, num_frames=81,
                seed=1, tiled=False,
                vace_video=batch["tar_video_key"][i:i+1].to(dtype=self.pipe.torch_dtype, device=self.device), vace_video_ref=batch["ref_video"][i:i+1].to(dtype=self.pipe.torch_dtype, device=self.pipe.device),
                vace_mask=batch["tar_video_key_mask"][i:i+1].to(dtype=self.pipe.torch_dtype, device=self.pipe.device), tar_video=batch['tar_video'][i:i+1].to(dtype=self.pipe.torch_dtype, device=self.pipe.device),
                ref_img_pil=ref_img_pil, inference=True,
            )
            save_dir = os.path.join(self.trainer.logger.save_dir, 'all_videos', self.trainer.logger._name, f'gs_{self.trainer.global_step}-{video_names[i]}_global_rank_{global_rank}.mp4')
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            save_video(video, save_dir, fps=16, quality=5)

            with open(save_dir.replace('.mp4', '.txt'), 'w') as f:
                f.write(f'{batch["prompt"][i]}\n')
                # f.write(f'instruct_prompt: {batch["instruct_prompt"][i]}\n')

            if ref_img_pil is not None:
                print('Save IP images...')
                img_path = save_dir.replace('.mp4', '.png')
                ref_img_pil.save(img_path)

        del video, batch
        self.pipe.load_models_to_device()       # offload model into cpu
        self.pipe.train()
        self.pipe.scheduler.set_timesteps(1000, training=True)


    # @torch.no_grad()
    # def on_train_batch_start(self, batch, batch_idx):
        
    #     if batch_idx == 0 and False:
    #         self.run_val_func(batch, batch_idx)
    #     else:
    #         pass

    # @torch.no_grad()
    # def on_train_batch_end(self, outputs, batch, batch_idx):
        
    #     # print logs
    #     if self.trainer.is_global_zero :
    #         print("accumulate_grad_batches =", self.trainer.accumulate_grad_batches)
    #         print("global_step =", self.trainer.global_step)
    #         print("PL accumulate_grad_batches:", self.trainer.accumulate_grad_batches)

    #     acc_batches = self.trainer.accumulate_grad_batches
    #     is_update_step = ((batch_idx + 1) % acc_batches == 0)
    #     init_sample = (acc_batches>1 and self.trainer.global_step==0 or self.trainer.global_step==1)

    #     if init_sample and not self.debug or (is_update_step and (self.trainer.global_step % self.log_video_steps == 0)):
    #         self.run_val_func(batch, batch_idx)
    #     else:
    #         pass

    @torch.no_grad()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.trainer.is_global_zero:
            print("accumulate_grad_batches =", self.trainer.accumulate_grad_batches)
            print("global_step =", self.trainer.global_step)

        acc_batches = self.trainer.accumulate_grad_batches
        
        # 判断是否为真正的 optimizer update step
        is_update_step = ((batch_idx + 1) % acc_batches == 0)
        
        # 修复优先级 bug，意图：梯度累积>1时，第0步或第1步触发初始采样
        init_sample = (
            acc_batches > 1 and 
            (self.trainer.global_step == 0)
        )
        should_log = (
            (init_sample and not self.debug) or 
            (is_update_step and (self.trainer.global_step % self.log_video_steps == 0))
        )

        if should_log:
            self.run_val_func(batch, batch_idx)


    def configure_optimizers(self):
        print(f'optimize model params: {self.train_architecture}')
        if self.train_architecture=='vace':
            trainable_modules = filter(lambda p: p.requires_grad, self.pipe.vace.parameters())
            all_module_names = [name for name, p in self.pipe.vace.named_parameters() if p.requires_grad]
        elif self.train_architecture=='lora':
            trainable_modules = filter(lambda p: p.requires_grad, self.pipe.vace.parameters())
            all_module_names = [name for name, p in self.pipe.vace.named_parameters() if p.requires_grad]
        elif self.train_architecture=='all_lora':
            import itertools
            trainable_modules_1 = filter(lambda p: p.requires_grad, self.pipe.vace.parameters())
            trainable_modules_2 = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
            trainable_modules = itertools.chain(trainable_modules_1, trainable_modules_2)
            all_module_names = [name for name, p in self.pipe.vace.named_parameters() if p.requires_grad] + [name for name, p in self.pipe.denoising_model().named_parameters() if p.requires_grad]
        elif self.train_architecture=='full':
            import itertools
            trainable_modules_1 = filter(lambda p: p.requires_grad, self.pipe.vace.parameters())
            trainable_modules_2 = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
            trainable_modules = itertools.chain(trainable_modules_1, trainable_modules_2)
            all_module_names = [name for name, p in self.pipe.vace.named_parameters() if p.requires_grad] + [name for name, p in self.pipe.denoising_model().named_parameters() if p.requires_grad]
        else:
            trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        # print(all_module_names)
        return optimizer


    def on_save_checkpoint(self, checkpoint):
        # checkpoint.clear()
        if self.trainer.global_rank == 0:
            # 提取你想保存的权重 (复用你原本的逻辑)
            keys_to_save = {}
            if self.train_architecture=='vace':
                trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.vace.named_parameters()))
            
                trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
                state_dict = self.pipe.vace.state_dict()
                vae_state_dict = {}
                for name, param in state_dict.items():
                    if name in trainable_param_names:
                        vae_state_dict[name] = param
                keys_to_save.update(vae_state_dict)
            elif self.train_architecture=='lora':

                trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.vace.named_parameters()))
                trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
                state_dict = self.pipe.vace.state_dict()
                lora_state_dict = {}
                for name, param in state_dict.items():
                    if name in trainable_param_names:
                        lora_state_dict[name] = param
                keys_to_save.update(lora_state_dict)
            elif self.train_architecture=='all_lora':
                # vace lora
                trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.vace.named_parameters()))
                trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
                state_dict = self.pipe.vace.state_dict()
                lora_state_dict = {}
                for name, param in state_dict.items():
                    if name in trainable_param_names:
                        lora_state_dict[name] = param
                keys_to_save.update(lora_state_dict)
                
                # dit lora
                trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
                trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
                state_dict = self.pipe.denoising_model().state_dict()
                denoising_state_dict = {}
                for name, param in state_dict.items():
                    if name in trainable_param_names:
                        denoising_state_dict[name] = param
                keys_to_save.update(denoising_state_dict)
            elif self.train_architecture=='full':
                # vace params
                trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.vace.named_parameters()))
                trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
                state_dict = self.pipe.vace.state_dict()
                lora_state_dict = {}
                for name, param in state_dict.items():
                    if name in trainable_param_names:
                        lora_state_dict[name] = param
                keys_to_save.update(lora_state_dict)
                
                # dit params
                trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
                trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
                state_dict = self.pipe.denoising_model().state_dict()
                denoising_state_dict = {}
                for name, param in state_dict.items():
                    if name in trainable_param_names:
                        denoising_state_dict[name] = param
                keys_to_save.update(denoising_state_dict)
            else:
                trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
            
                trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
                state_dict = self.pipe.denoising_model().state_dict()
                lora_state_dict = {}
                for name, param in state_dict.items():
                    if name in trainable_param_names:
                        lora_state_dict[name] = param
                keys_to_save.update(lora_state_dict)


            save_path = os.path.join(self.trainer.checkpoint_callback.dirpath,
                                    f"lora_weights_wan-epoch={self.current_epoch}-step={self.global_step}.ckpt")
            
            # save float32
            for k, v in list(keys_to_save.items()):
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        keys_to_save[k] = v.detach().to(torch.float32).cpu()
                    else:
                        keys_to_save[k] = v.detach().cpu()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(keys_to_save, save_path)
            print(f"Full Lightning checkpoint saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    # Dataset information
    parser.add_argument(
        "--aug_mode",
        type=str,
        default='basic',
        help='aoss_client.client user config path',
    )
    parser.add_argument(
        "--single_task",
        type=str,
        default=None,
        help='single task name..'
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="video height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="video width.",
    )
    parser.add_argument("--fps", type=int, default=16, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )

    # Training
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="world_size",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="world_size",
    )
    parser.add_argument(
        "--every_n_train_steps",
        type=int,
        default=1000,
        help="ckpt logs.",
    )
    parser.add_argument(
        "--log_video_steps",
        type=int,
        default=500,
        help="training video logs.",
    )
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="Log loss every N steps")
    parser.add_argument(
        "--train_from_sratch",
        default=False,
        action="store_true",
        help="ckpt load methods",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="train_1",
        help="wandb project name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="wandb run name.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--add_mask_loss",
        default=False,
        action="store_true",
        help="ckpt load methods",
    )
    parser.add_argument(
        "--resume_ckpt_folder",
        type=str,
        default=None,
        help="Path to resume the training process...",
    )


    # Model path..
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3", "ddp"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "all_lora", "full", "vace"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


def train(args):
    """
    Baseline settings: with only MSE loss
    """
    # ------------ 1. Load dataset and model
    use_contrast_loss = False
    use_mse_loss = False
    use_attnscore_loss = False
    debug_ornot = args.debug
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
        log_video_steps=args.log_video_steps,
        train_from_sratch=args.train_from_sratch,
        single_task=args.single_task,
        height=args.height, width=args.width,
        num_frames=args.num_frames,
        train_batch_size=args.train_batch_size,
        use_contrast_loss=use_contrast_loss,
        use_mse_loss=use_mse_loss,
        use_attnscore_loss=use_attnscore_loss,
        debug_ornot=debug_ornot,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(
        # dirpath=args.output_path,
        filename="wan_deepspeed_folder-{epoch}-{step}",
        save_top_k=-1,  
        every_n_train_steps=args.every_n_train_steps,        
    )
    # lr_monitor = pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step")

    from pytorch_lightning.loggers import WandbLogger

    wandb_logger = WandbLogger(
        name=args.run_name,
        project=args.project_name,
        log_model=False,            # 可选：保存模型结构和权重
        save_dir=args.output_path,
        version=f'{args.run_name}',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        num_nodes=args.num_nodes,
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        sync_batchnorm=True,
        log_every_n_steps=args.log_every_n_steps,
    )
    if args.resume_ckpt_folder is not None:
        trainer.fit(model, ckpt_path=args.resume_ckpt_folder)
    else:
        trainer.fit(model)



if __name__ == '__main__':
    args = parse_args()
    train(args)

