#!/bin/bash
export WANDB_API_KEY="<your_api_key>"

NNODES=1
WORLD_SIZE=8 #should be total gpu numbers, e.g. 16 for 2 nodes(8 gpus each)
MACHINE_RANK=0

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --machine_rank $MACHINE_RANK \
"

SCRIPT="train_RORem_lcm.py"

SCRIPT_ARGS=" \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --output_dir experiment/RORem_LCM \
    --pretrained_teacher_model diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --pretrained_teacher_unet xxx \
    --report_to wandb \
    --resolution 512 \
    --tracker_project_name LCM-Finetune\
    --learning_rate 5e-5 \
    --max_train_steps 20000 \
    --validation_steps 2000 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 10 \
    --seed 4 \
    --mixed_precision fp16 \
    --enable_xformers_memory_efficient_attention \
    "


export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
echo $CMD
$CMD

