import os
import argparse
from collections import OrderedDict

import torch
from safetensors.torch import save_file

# Utility from your project
from diffsynth.models.utils import load_state_dict


def unwrap_checkpoint(obj):
    """
    Compatible with:
    1) A raw state_dict
    2) A Lightning/checkpoint dict that wraps state_dict
    """
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def strip_wrappers(key: str) -> str:
    """
    Strip common outer prefixes.
    """
    prefixes = [
        "state_dict.",
        "model.",
        "module.",
        "_forward_module.",
    ]
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def convert_one_key(old_key: str) -> str | None:
    """
    Convert diffstn/custom Wan-VACE LoRA keys
    into diffusers-readable keys (with `transformer.` prefix).
    """

    k = strip_wrappers(old_key)

    # Keep LoRA parameters only
    keep = (
        ".lora_A." in k
        or ".lora_B." in k
        or k.endswith(".alpha")
        or ".lora_magnitude_vector" in k
    )
    if not keep:
        return None

    # Old keys are in PEFT internal format and include `.default`.
    # Remove adapter name before saving for the diffusers loader.
    k = k.replace(".lora_A.default.weight", ".lora_A.weight")
    k = k.replace(".lora_B.default.weight", ".lora_B.weight")
    k = k.replace(".lora_A.default.bias", ".lora_A.bias")
    k = k.replace(".lora_B.default.bias", ".lora_B.bias")

    # Core rename rules: aligned with official Wan module mapping
    rename_rules = [
        ("self_attn.q", "attn1.to_q"),
        ("self_attn.k", "attn1.to_k"),
        ("self_attn.v", "attn1.to_v"),
        ("self_attn.o", "attn1.to_out.0"),
        ("cross_attn.q", "attn2.to_q"),
        ("cross_attn.k", "attn2.to_k"),
        ("cross_attn.v", "attn2.to_v"),
        ("cross_attn.o", "attn2.to_out.0"),
        ("ffn.0", "ffn.net.0.proj"),
        ("ffn.2", "ffn.net.2"),
    ]
    for src, dst in rename_rules:
        k = k.replace(src, dst)

    # Important: convert to the prefix style used by diffusers pipeline save/load
    if not k.startswith("transformer."):
        k = "transformer." + k

    return k


def convert_state_dict(old_sd: dict) -> OrderedDict:
    new_sd = OrderedDict()
    for k, v in old_sd.items():
        new_k = convert_one_key(k)
        if new_k is None:
            continue
        new_sd[new_k] = v.contiguous() if isinstance(v, torch.Tensor) else v
    return new_sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="all_ckpts/ReCo_ori_rank128-2025_m12_version.ckpt", help="Source LoRA checkpoint / safetensors")
    parser.add_argument("--dst", type=str, default="all_ckpts/lora_diffusers.safetensors", help="Output safetensors path")
    args = parser.parse_args()

    raw = load_state_dict(args.src, torch_dtype=None)
    raw = unwrap_checkpoint(raw)

    new_sd = convert_state_dict(raw)

    if len(new_sd) == 0:
        raise RuntimeError(
            "No parameters remain after conversion. Please print the original keys first and check whether they still have an extra wrapper prefix, or whether the source file is not a LoRA checkpoint."
        )

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    save_file(dict(new_sd), args.dst)

    print(f"[OK] saved to: {args.dst}")
    print(f"[OK] num keys: {len(new_sd)}")
    print("[Preview] first 20 keys:")
    for i, k in enumerate(list(new_sd.keys())[:20]):
        print(f"  {i+1:02d}. {k}")


if __name__ == "__main__":
    main()