import os
import time
import torch
from torch.utils.flop_counter import FlopCounterMode
from diffusers import WanVACEPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

torch.backends.cudnn.benchmark = True

model_id = "/mnt/zhongwei/subapp/t2v_models/Wan-AI/Wan2.1-VACE-1.3B-diffusers"
flow_shift = 5.0
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    vae=vae,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
pipe.to("cuda")

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# ================================ Load LoRA ================================
lora_path = "lora_diffusers.safetensors"
lora_path = os.path.abspath(lora_path)
if not os.path.isfile(lora_path):
    raise FileNotFoundError(f"LoRA file not found: {lora_path}")

lora_dir = os.path.dirname(lora_path) or "."
weight_name = os.path.basename(lora_path)

pipe.load_lora_weights(
    lora_dir,
    weight_name=weight_name,
    adapter_name="default",
)
pipe.set_adapters(["default"], [1.0])


# ================================ Inference ================================
def prepare_video_concat_gray_padded_mask(video_path: str, height: int, width: int) -> tuple[list, list]:
    import decord
    from PIL import Image

    vr = decord.VideoReader(video_path)
    frames = []
    masks = []

    pad_color = (128, 128, 128)

    for frame_arr in vr:
        frame = frame_arr.asnumpy() if hasattr(frame_arr, "asnumpy") else frame_arr
        img = Image.fromarray(frame).resize((width, height))

        padded = Image.new("RGB", (width * 2, height), pad_color)
        padded.paste(img, (0, 0))

        mask_img = Image.new("L", (width * 2, height), 255)
        mask_img.paste(0, (0, 0, width, height))

        frames.append(padded)
        masks.append(mask_img)

    return frames, masks


video_path = "video_path.mp4"
prompt = "prompt.txt"

if not os.path.isfile(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

height, width = 480, 832
video_frames, mask_frames = prepare_video_concat_gray_padded_mask(video_path, height, width)
num_frames = min(len(video_frames), 81)
ref_images = None

# ================================ Benchmark ================================
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    # warmup: let CUDA kernels compile/cache (not counted in timing)
    _ = pipe(
        prompt=prompt,
        video=video_frames,
        mask=mask_frames,
        reference_images=ref_images,
        num_frames=num_frames,
        height=height,
        width=width * 2,
        num_inference_steps=1,
        guidance_scale=5.0,
        conditioning_scale=1.0,
        generator=torch.Generator().manual_seed(42),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    num_inference_steps = 30
    flop_counter = FlopCounterMode(display=False)
    t_start = time.perf_counter()

    with flop_counter:
        out = pipe(
            prompt=prompt,
            video=video_frames,
            mask=mask_frames,
            reference_images=ref_images,
            num_frames=num_frames,
            height=height,
            width=width * 2,
            num_inference_steps=num_inference_steps,
            guidance_scale=5.0,
            conditioning_scale=1.0,
            generator=torch.Generator().manual_seed(42),
        ).frames[0]

    torch.cuda.synchronize()
    t_end = time.perf_counter()

total_flops = flop_counter.get_total_flops()
per_step_flops = total_flops / num_inference_steps
peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
inference_time = t_end - t_start

def fmt_flops(f):
    if f >= 1e15:
        return f"{f / 1e15:.2f} PFLOPs"
    if f >= 1e12:
        return f"{f / 1e12:.2f} TFLOPs"
    if f >= 1e9:
        return f"{f / 1e9:.2f} GFLOPs"
    return f"{f / 1e6:.2f} MFLOPs"

print(f"\n{'=' * 50}")
print(f"  Core inference time : {inference_time:.2f} s")
print(f"  Peak GPU memory     : {peak_mem_gb:.2f} GB")
print(f"  Total FLOPs         : {fmt_flops(total_flops)}")
print(f"  Per-step FLOPs      : {fmt_flops(per_step_flops)}")
print(f"  Num frames          : {num_frames}")
print(f"  Num inference steps : {num_inference_steps}")
print(f"  Resolution          : {height}x{width * 2}")
print(f"{'=' * 50}\n")

export_to_video(out, "outputs/output_2_30.mp4", fps=16)

