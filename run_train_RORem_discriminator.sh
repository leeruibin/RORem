#!/bin/bash
export WANDB_API_KEY="<your_api_key>"

LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --machine_rank $MACHINE_RANK \
"


SCRIPT="train_discrinimator_with_enhance_data_dilate.py"

SCRIPT_ARGS=" \
    --train_batch_size 32 \
    --rank 4 \
    --output_dir /home/ubuntu/sagemaker/liruibin/CosXL_edit/experiment/RefineDiscriminator_synthesis_dilate_bs32 \
    --max_train_steps 15000 \
    --pretrained_model_name_or_path diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --validation_steps 1000 \
    --mixed_precision fp16 \
    --learning_rate 1e-5 \
    --seed 20240421 \
    --report_to wandb \
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
echo $CMD
srun $CMD
