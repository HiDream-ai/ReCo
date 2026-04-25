#!/usr/bin/env bash

# cd DiffSynth-Studio
# pip install -e .
# cd ..

# ========== Env (W&B) ==========
export WANDB_API_KEY=""
export WANDB_PROJECT="QA_GEN_JOINT"
export WANDB_MODE="offline"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export GPU_NUM=8
export NUM_NODES=2              # 1 or 2
export TOKENIZERS_PARALLELISM=false  # Linux/macOS

# ========== Config ==========
DIT_PATH="Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors,\
VACE/models/VACE-Wan2.1-1.3B/models_t5_umt5-xxl-enc-bf16.pth,\
VACE/models/VACE-Wan2.1-1.3B/Wan2.1_VAE.pth"

# training
world_size=$((GPU_NUM * NUM_NODES))
training_strategy='deepspeed_stage_2'           # or ddp

# logs dir
OUTPUT_PATH="./all_results/train_multitask_add_kiwi_ref_data"
script_name=$(basename "$0" .sh)
log_dir="./all_results/logs/$script_name"
project_name="train_video_edit"
run_name="train_add_kiwi_ref_data_2node"  # NO CHANGE..
log_file="$log_dir/$run_name/train.log"
mkdir -p "$log_dir/$run_name"
mkdir -p "$OUTPUT_PATH"

# ========== Run ==========
torchrun --nnodes=$NUM_NODES \
    --nproc_per_node=$GPU_NUM \
    --rdzv_id=distributed_alldata \
    --rdzv_backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR \
    train_multitask_add_kiwi_ref_data.py --train_architecture all_lora \
    --num_nodes $NUM_NODES \
    --training_strategy $training_strategy \
    --every_n_train_steps 250 \
    --log_video_steps 250 \
    --train_batch_size 1 \
    --world_size $world_size \
    --dataloader_num_workers 1 \
    --output_path "$OUTPUT_PATH" \
    --dit_path "$DIT_PATH" \
    --max_epochs 10 \
    --learning_rate 2e-5 \
    --accumulate_grad_batches 2 \
    --log_every_n_steps 10 \
    --use_gradient_checkpointing \
    --lora_rank "256" --lora_alpha "256" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --project_name $project_name \
    --run_name $run_name  2>&1 | tee $log_file

    # 如需加载已有 LoRA：
    # --pretrained_lora_path "all_results/train_runs/.../checkpoints/xxx.ckpt"
    # --resume_ckpt_folder "all_results/train_runs/..../checkpoints/wan_deepspeed_folder-epoch=0-step=7000.ckpt" 2>&1 | tee $log_file


