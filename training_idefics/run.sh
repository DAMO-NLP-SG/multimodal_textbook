#!/bin/bash


deepspeed_path=$(which deepspeed)
if [ -z "$deepspeed_path" ]; then
    echo "deepspeed not found. Please make sure it is installed and added to your PATH."
    exit 1
fi


base_path="/mnt/data/zh/zwq"

model_name_or_path="$base_path/model2/idefics2-8b-base"
deepspeed_config="$base_path/interleaved_LMM/training/zero3.json"
data_path="$base_path/interleaved_dataset/ours_textbook_video_clip_format.json"
base_folder="$base_path/interleaved_LMM"
output_dir="$base_path/training_output/output"


$deepspeed_path --master_port=61000 training/training.py \
    --model_name_or_path "$model_name_or_path" \
    --deepspeed "$deepspeed_config" \
    --data_path "$data_path" \
    --base_folder "$base_folder" \
    --freeze_backbone "False" \
    --bf16 "True" \
    --output_dir "$output_dir" \
    --num_train_epochs "1" \
    --per_device_train_batch_size "2" \
    --per_device_eval_batch_size "4" \
    --gradient_accumulation_steps "128" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "80" \
    --save_total_limit "10" \
    --learning_rate "1e-6" \
    --weight_decay "0." \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --logging_steps "1" \
    --tf32 "True" \
    --model_max_length "4096" \
    --gradient_checkpointing "True" \
    --dataloader_num_workers "8" \
    --h100 "False" \
    --train_scratch "False" \
    --scratch_model_path "/model/idefics2-8b-from-mistral-siglip" \
    --high_res "False" \
    --add_ocr "False"

