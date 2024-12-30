#!/bin/bash

pip uninstall transfomrers
pip install transformers==4.40.0

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=61000
ARG_RANK=0

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"


# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=interleaved_LMM
BASE_FOLDER="/mnt/workspace/interleaved_LMM"
MODEL_NAME_OR_PATH="/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base"
DEEPSPEED_CONFIG="/mnt/workspace/interleaved_LMM/training/zero3.json"
DATA_PATH="/mnt/group_data/zwq_data/interleaved_dataset/clip_level_sample_2k/llava_no_contact.json"
OUTPUT_DIR="/mnt/workspace/zwq_data/training_output/595kllava_lr1e5_epoch1_gbatch1024_context1024_1node"

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    training/training.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --deepspeed $DEEPSPEED_CONFIG \
    --data_path $DATA_PATH \
    --base_folder $BASE_FOLDER \
    --freeze_backbone "False" \
    --bf16 "True" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs "1" \
    --per_device_train_batch_size "8" \
    --per_device_eval_batch_size "4" \
    --gradient_accumulation_steps "16" \
    --save_strategy "steps" \
    --save_steps "100" \
    --save_total_limit "10" \
    --learning_rate "1e-6" \
    --weight_decay "0." \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --logging_steps "1" \
    --tf32 "True" \
    --model_max_length "1024" \
    --gradient_checkpointing "True" \
    --dataloader_num_workers "8"