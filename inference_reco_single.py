import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video
from PIL import Image
from tqdm import tqdm
import os
from diffsynth.models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model

import decord
import numpy as np
import random
import torch.nn.functional as F
import argparse



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)



def add_lora_to_model(model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
    # Add LoRA to UNet
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



def sample_and_resize_video(tar_video, num_frames):
    num_f, h,w,c = tar_video.shape

    # add or remove sample frames
    if num_f > num_frames:
        begin_idx = random.randint(0, num_f - num_frames)
        end_idx = begin_idx + num_frames
        tar_video = tar_video[begin_idx:end_idx]

    elif num_f <= num_frames:
        pad_len = num_frames - num_f
        last_frame = tar_video[-1:]

        tar_video = np.concatenate([tar_video, np.repeat(last_frame, pad_len, axis=0)], axis=0)

    tar_video = F.interpolate(torch.from_numpy(tar_video).float().permute(0,3,1,2), (480,832)).permute(0,2,3,1)

    return tar_video


def get_batch_from_video(video_path, ref_img, prompt, ip_img_path, h, w, f):
        
    # 1. read video and interpolate
    vr = decord.VideoReader(video_path)
    video_name = os.path.basename(video_path).replace('.mp4', '')
    video = np.array(vr.get_batch(np.arange(0, len(vr))).asnumpy())
    ref_video = sample_and_resize_video(video, f)
    ref_video = (ref_video / 255.) *2 - 1                           # convert into [-1, 1]

    rigth_tensor = torch.zeros((f, h, w//2, 3), dtype=torch.float32)
    tar_video_key_mask = torch.zeros((f, h, w, 3), dtype=torch.float32)

    if ref_img is not None:
        ref_img_ts = torch.from_numpy(np.array(Image.open(ref_img).convert("RGB").resize((w//2, h)))).float()  # [h, w//2, 3]
        rigth_tensor[0, :, :, :] = (ref_img_ts / 255.) * 2 - 1
        tar_video_key_mask[1:, :, w//2:, :] = 1.0
    else:
        tar_video_key_mask[:, :, w//2:, :] = 1.0

    tar_video_key = torch.concat([ref_video, rigth_tensor], dim=2)  # [f, h, w*2, 3]
    tar_video_key = tar_video_key * (1-tar_video_key_mask)

    return {
        "tar_video_key": tar_video_key.permute(3, 0, 1, 2).unsqueeze(0),  # [3, f, h, w*2]
        "tar_video_key_mask": tar_video_key_mask.permute(3, 0, 1, 2).unsqueeze(0),  # [3, f, h, w*2]
        "ref_video": torch.zeros_like(tar_video_key.permute(3,0,1,2)).unsqueeze(0),  # [f, h, w*2, 3]
        "tar_video": torch.zeros_like(tar_video_key.permute(3,0,1,2)).unsqueeze(0),  # [f, h, w*2, 3]
        "prompt": [prompt],
        "video_name": [video_name],
        "ip_img_path": [ip_img_path],
    }



def get_processed_ref_img(ref_img_path, height, width):

    if ref_img_path is not None:
        ref_img_pil = Image.open(ref_img_path).convert("RGB")

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



from typing import List, Dict, Optional

def parse_instruction_file(
    file_path: str,
    encoding: str = "utf-8",
) -> List[Dict[str, str]]:
    """
    读取形如：
        855029-hd_1920_1080_30fps.mp4: Add ... | asserts/ref_img_path/rabbit_2.png
    的文本文件并解析为字典列表：
        [{"src_video_path": ..., "instructed_prompt": ..., "ref_img_path": ...}, ...]
    
    规则：
    - 允许行首以 # 开头作为注释，或空行，均跳过
    - 仅使用第一处冒号分割出 video 与其余部分
    - 使用 ' | '（两侧可有可无多余空格）分割出 prompt 与 ref_img_path
    - 若缺少 ref_img_path ""（空字符串）
    """
    results: List[Dict[str, str]] = []
    with open(file_path, "r", encoding=encoding) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # 1) 拆出 video 与其余部分（只按第一个冒号切）
            if ":" not in line:
                raise ValueError(f"[line {lineno}] 格式错误：缺少冒号 ':' —— {raw!r}")
            video, rest = line.split(":", 1)
            video = video.strip()
            rest = rest.strip()

            if not video:
                raise ValueError(f"[line {lineno}] 格式错误：src_video_path 为空 —— {raw!r}")

            # 2) 拆出 prompt 与 ip（'|' 可选）
            ref_img_path = None
            if "|" in rest:
                prompt, ip = rest.split("|", 1)
                prompt = prompt.strip()
                ref_img_path = ip.strip()
            else:
                prompt = rest.strip()  # 允许没有 ip 的行

            if not prompt:
                raise ValueError(f"[line {lineno}] 格式错误：instructed_prompt 为空 —— {raw!r}")

            # 3) 规范化 ref_img_path
            if ref_img_path:
                import os
                ref_img_path = os.path.normpath(ref_img_path)

            results.append({
                "src_video_path": video,
                "instructed_prompt": prompt,
                "ref_img_path": ref_img_path,
            })

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_txt_file_name", type=str, default="assets/remove_test_given_first_frame.txt")
    parser.add_argument("--task_name", type=str, default="remove_wf", choices=["remove", "replace", "add", "style", \
                                                                            "remove_wf", "replace_wf", "add_wf", "style_wf",])
    parser.add_argument("--base_video_folder", type=str, default="assets/test_videos")
    parser.add_argument("--process_id", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--base_wan_folder", type=str, default="./Wan-AI")
    parser.add_argument("--lora_ckpt", type=str, default="all_ckpts/2026_01_16_v1_release_preview.ckpt")
    args = parser.parse_args()

    # ============================ 1. Pre-Process ckpts ==============
    seed_everything(2025)
    use_lora = True
    use_dit_lora = True
    lora_rank = 128
    lora_alpha = 128
    lora_target_modules="q,k,v,o,ffn.0,ffn.2"

    # ============================ 2. Get dataset =================
    base_video_folder = args.base_video_folder
    file_name = args.test_txt_file_name
    all_video_list = parse_instruction_file(file_name)
    if args.multi_gpu:
        all_video_list = all_video_list[args.process_id::8*args.num_nodes]

    # ============================ 3. Load model ==================
    lora_ckpt = args.lora_ckpt
    save_path = f"all_results/single_test"

    ckpt_list = [
            f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
            f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth",
        ]

    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        ckpt_list,
        torch_dtype=torch.float16,
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.float16, device="mps")

    if use_lora:
        # Load LoRA weights
        add_lora_to_model(
        pipe.vace,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,                      # alpha/r * BA
        lora_target_modules=lora_target_modules,
        init_lora_weights="kaiming",
        pretrained_lora_path=lora_ckpt,
        )
    if use_dit_lora:
        add_lora_to_model(
        pipe.denoising_model(),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,                      # alpha/r * BA
        lora_target_modules=lora_target_modules,
        init_lora_weights="kaiming",
        pretrained_lora_path=lora_ckpt,
        )
    
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.eval()


    # ==================== 4. Start iter ================
    for j,video_dict in tqdm(enumerate(all_video_list), total=len(all_video_list)):

        prompt = video_dict['instructed_prompt']
        video_base_name = video_dict['src_video_path']
        ip_path = None

        # ------------- 2. inference model
        video_path = os.path.join(base_video_folder, video_base_name)
        ref_img = video_dict['ref_img_path']

        batch = get_batch_from_video(video_path, ref_img, prompt, ip_path, 480, 832*2, 81)

        # Depth video + Reference image -> Video
        global_rank = 1
        device = pipe.device
        video_names = batch["video_name"]
        task_name = args.task_name
        # dataset_type = batch["dataset_type"]
        b,c,f,h,w = batch["tar_video_key"].shape

        with torch.no_grad():
            # Depth video + Reference image -> Video
            ip_img_pil = get_processed_ref_img(batch["ip_img_path"][0], 480, 832)    # bs = 1, NOTE: input ip_images
            negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

            prompt_name = batch["prompt"][0].replace(' ', '_').replace('.', '').replace(',','').replace(':',' ')
            vide_save_name = video_base_name.replace('.mp4', '')
            video_save_name = f'{vide_save_name}_{prompt_name[:80]}.mp4'
            save_dir = os.path.join(save_path, args.task_name, video_save_name)
            if os.path.exists(save_dir):
                continue

            video = pipe(
                prompt=batch["prompt"][0],
                negative_prompt=negative_prompt,
                num_inference_steps=50,
                height=h, width=w, num_frames=81,
                seed=1, tiled=False,
                vace_video=batch["tar_video_key"][:1].to(dtype=pipe.torch_dtype, device=device), vace_video_ref=batch["ref_video"][:1].to(dtype=pipe.torch_dtype, device=device),
                vace_mask=batch["tar_video_key_mask"][:1].to(dtype=pipe.torch_dtype, device=pipe.device), tar_video=batch['tar_video'][:1].to(dtype=pipe.torch_dtype, device=device),
                ref_img_pil=ip_img_pil, inference=True,
            )


            os.makedirs(os.path.dirname(save_dir), exist_ok=True) 
            save_video(video, save_dir, fps=16, quality=5)

            with open(save_dir.replace('.mp4', '.txt'), 'w') as f:
                f.write(f'prompt: {batch["prompt"][0]}\n')
                # f.write(f'instruct_prompt: {batch["instruct_prompt"][i]}\n')

            if ip_img_pil is not None:
                img_path = save_dir.replace('.mp4', '.png')
                ip_img_pil.save(img_path)
            



