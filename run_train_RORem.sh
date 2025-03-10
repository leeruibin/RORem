export WANDB_API_KEY=<your_api_key>

NNODES=1
WORLD_SIZE=8
MACHINE_RANK=0

export LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --machine_rank $MACHINE_RANK \
    "
# if you use multiple nodes to train to model, add the following args into the LAUMCHER
# --main_process_ip "$MASTER_ADDR" \
# --main_process_port $MASTER_PORT \

BASE_PATH="xxx" # path to your dataset folder

JSON_FILES="\
${BASE_PATH}/Final_open_RORem/meta.json \
"

# for multiple json file you can set as 
# JSON_FILES="\
# ${BASE_PATH}/Final_open_RORem/meta.json \
# ${BASE_PATH}/yourdata1/meta.json \
# ${BASE_PATH}/yourdata2/meta.json \
# "

OUTPUT_FOLDER=<your_path_to_save_checkpoint>

SCRIPT_ARGS=" \
    --train_batch_size 16 \
    --output_dir $OUTPUT_FOLDER \
    --meta_path $JSON_FILES \
    --max_train_steps 50000 \
    --random_flip \
    --resolution 512 \
    --pretrained_model_name_or_path diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --mixed_precision fp16 \
    --checkpoints_total_limit 5 \
    --checkpointing_steps 5000 \
    --learning_rate 5e-5 \
    --validation_steps 2000 \
    --seed 4 \
    --report_to wandb \
    "

SCRIPT=train_RORem.py

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
bash $CMD

