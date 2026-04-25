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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_lora_to_model(model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                      init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
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
        if param.requires_grad:
            param.data = param.to(torch.float32)

    if pretrained_lora_path is not None:
        state_dict = load_state_dict(pretrained_lora_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. "
              f"{num_unexpected_keys} parameters are unexpected.")


def sample_and_resize_video(tar_video, num_frames):
    num_f, h, w, c = tar_video.shape

    if num_f > num_frames:
        begin_idx = random.randint(0, num_f - num_frames)
        end_idx = begin_idx + num_frames
        tar_video = tar_video[begin_idx:end_idx]
    elif num_f <= num_frames:
        pad_len = num_frames - num_f
        last_frame = tar_video[-1:]
        tar_video = np.concatenate([tar_video, np.repeat(last_frame, pad_len, axis=0)], axis=0)

    tar_video = F.interpolate(
        torch.from_numpy(tar_video).float().permute(0, 3, 1, 2), (480, 832)
    ).permute(0, 2, 3, 1)
    return tar_video


def get_batch_from_video(video_path, prompt, h, w, f, first_frame_path=None):
    """
    Build an inference batch.

    first_frame_path: Optional path to a "first-frame condition" image.
        - If provided: the image is embedded into frame 0 of the right half of tar_video_key,
          and tar_video_key_mask only masks frames from 1 onward (the model treats frame 0 as known
          and generates subsequent frames).
        - If not provided: tar_video_key_mask masks the right half for all frames (the model freely
          generates all frames).

    Note: the IP image is independent of this; it is passed into the pipeline via
    get_processed_ref_img + ref_img_pil.
    """
    vr = decord.VideoReader(video_path)
    video_name = os.path.basename(video_path).replace('.mp4', '')
    video = np.array(vr.get_batch(np.arange(0, len(vr))).asnumpy())
    src_video = sample_and_resize_video(video, f)
    src_video = (src_video / 255.) * 2 - 1         # [-1, 1]

    right_tensor = torch.zeros((f, h, w // 2, 3), dtype=torch.float32)
    tar_video_key_mask = torch.zeros((f, h, w, 3), dtype=torch.float32)

    if first_frame_path is not None:
        first_frame_ts = torch.from_numpy(
            np.array(Image.open(first_frame_path).convert("RGB").resize((w // 2, h)))
        ).float()                                           # [h, w//2, 3]
        right_tensor[0, :, :, :] = (first_frame_ts / 255.) * 2 - 1
        tar_video_key_mask[1:, :, w // 2:, :] = 1.0       # frame 0 is known; mask the rest
    else:
        tar_video_key_mask[:, :, w // 2:, :] = 1.0        # mask all; free generation

    tar_video_key = torch.concat([src_video, right_tensor], dim=2)     # [f, h, w*2, 3]
    tar_video_key = tar_video_key * (1 - tar_video_key_mask)

    return {
        "tar_video_key": tar_video_key.permute(3, 0, 1, 2).unsqueeze(0),           # [1, 3, f, h, w*2]
        "tar_video_key_mask": tar_video_key_mask.permute(3, 0, 1, 2).unsqueeze(0), # [1, 3, f, h, w*2]
        "ref_video": torch.zeros_like(tar_video_key.permute(3, 0, 1, 2)).unsqueeze(0),
        "tar_video": torch.zeros_like(tar_video_key.permute(3, 0, 1, 2)).unsqueeze(0),
        "prompt": [prompt],
        "video_name": [video_name],
    }


def get_processed_ref_img(ip_img_path, height, width):
    """
    Process the IP image (fully independent from the first-frame condition).
    Keep consistent with training-time get_processed_ref_img (no random dropout at inference).
    Returns a concatenated PIL Image in the "left white / right image" layout, or None if not provided.
    """
    if ip_img_path is None:
        return None

    ref_img_pil = Image.open(ip_img_path).convert("RGB")
    ref_width, ref_height = ref_img_pil.size
    canvas_height, canvas_width = height, width

    scale = min(canvas_height / ref_height, canvas_width / ref_width)
    new_height = int(ref_height * scale)
    new_width = int(ref_width * scale)

    resized_pil = ref_img_pil.resize((new_width, new_height), Image.LANCZOS)
    white_canvas_pil = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

    top = (canvas_height - new_height) // 2
    left = (canvas_width - new_width) // 2
    white_canvas_pil.paste(resized_pil, (left, top))
    white_canvas_np = np.array(white_canvas_pil)

    ref_img_pil = Image.fromarray(
        np.concatenate([np.ones_like(white_canvas_np) * 255, white_canvas_np], axis=1)
    )

    return ref_img_pil


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True,
                        help="Input source video path, e.g. assets/test_videos/my_video.mp4")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt")
    parser.add_argument("--first_frame_path", type=str, default=None,
                        help="[First-frame condition] Reference image path. It will be embedded into "
                             "frame 0 of the right half of VACE tar_video_key to constrain the first "
                             "frame content. If not provided, the model freely generates all frames.")
    parser.add_argument("--ip_img_path", type=str, default=None,
                        help="[IP image] Reference image path. Passed into the pipeline as an IP-adapter "
                             "condition (ref_img_pil) to control style/appearance. Fully independent from "
                             "the first-frame condition; can be used alone or together.")
    parser.add_argument("--save_path", type=str, default="all_results/single_test",
                        help="Output directory for inference results")
    parser.add_argument("--base_wan_folder", type=str, default="./Wan-AI",
                        help="Base Wan model folder")
    parser.add_argument("--lora_ckpt", type=str, default="all_ckpts/ReCo_ref_rank256-2026_m4_version.ckpt",
                        help="Path to the LoRA checkpoint")
    args = parser.parse_args()

    # ============================ 1. Basic config ============================
    seed_everything(2025)
    use_lora = True
    use_dit_lora = True
    lora_rank = 256
    lora_alpha = 256
    lora_target_modules = "q,k,v,o,ffn.0,ffn.2"

    # ============================ 2. Load models ============================
    ckpt_list = [
        f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
        f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        f"{args.base_wan_folder}/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth",
    ]

    model_manager = ModelManager(device="cpu")
    model_manager.load_models(ckpt_list, torch_dtype=torch.bfloat16)
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")

    if use_lora:
        add_lora_to_model(
            pipe.vace,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights="kaiming",
            pretrained_lora_path=args.lora_ckpt,
        )
    if use_dit_lora:
        add_lora_to_model(
            pipe.denoising_model(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            init_lora_weights="kaiming",
            pretrained_lora_path=args.lora_ckpt,
        )

    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.eval()

    # ============================ 3. Prepare batch ============================
    h, w, num_frames = 480, 832 * 2, 81

    batch = get_batch_from_video(
        args.video_path, args.prompt, h, w, num_frames,
        first_frame_path=args.first_frame_path,
    )

    b, c, f, bh, bw = batch["tar_video_key"].shape
    device = pipe.device

    # ============================ 4. Inference ============================
    with torch.no_grad():
        # IP image: independent from the first-frame condition; processed by get_processed_ref_img
        # and passed into the pipeline as ref_img_pil.
        ref_img_pil = get_processed_ref_img(args.ip_img_path, 480, 832)

        negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, images, static, overall gray, "
            "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
            "three legs, many people in the background, walking backwards"
        )

        video = pipe(
            prompt=batch["prompt"][0],
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            height=bh, width=bw, num_frames=81,
            seed=1, tiled=False,
            vace_video=batch["tar_video_key"][0:1].to(dtype=pipe.torch_dtype, device=device),
            vace_video_ref=batch["ref_video"][0:1].to(dtype=pipe.torch_dtype, device=device),
            vace_mask=batch["tar_video_key_mask"][0:1].to(dtype=pipe.torch_dtype, device=device),
            tar_video=batch["tar_video"][0:1].to(dtype=pipe.torch_dtype, device=device),
            ref_img_pil=ref_img_pil, inference=True,
        )

    # ============================ 5. Save outputs ============================
    video_name = batch["video_name"][0]
    prompt_slug = args.prompt.replace(' ', '_').replace('.', '').replace(',', '').replace(':', ' ')
    video_save_name = f"{video_name}_{prompt_slug[:80]}.mp4"
    save_dir = os.path.join(args.save_path, video_save_name)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    save_video(video, save_dir, fps=16, quality=5)
    print(f"Video saved to: {save_dir}")

    with open(save_dir.replace('.mp4', '.txt'), 'w') as f:
        f.write(f'prompt: {args.prompt}\n')
        if args.first_frame_path:
            f.write(f'first_frame_path: {args.first_frame_path}\n')
        if args.ip_img_path:
            f.write(f'ip_img_path: {args.ip_img_path}\n')

    if ref_img_pil is not None:
        img_path = save_dir.replace('.mp4', '_ip.png')
        ref_img_pil.save(img_path)
        print(f"IP image saved to: {img_path}")
