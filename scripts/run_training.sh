deepspeed_path=$(which deepspeed)
if [ -z "$deepspeed_path" ]; then
    echo "deepspeed not found. Please make sure it is installed and added to your PATH."
    exit 1
fi

$deepspeed_path --master_port=61000 ./llava/train/train_interleaved_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/workspace/zwq_data/model/llava-v1.5-7b-sft \
    --version plain \
    --data_type "obelisc" \
    --data_path /mnt/workspace/multimodal_textbook/example_data/textbook_sample_100.json \
    --image_folder None \
    --vision_tower /mnt/workspace/zwq_data/model/openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter False \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --num_train_epochs 1 \
    --output_dir /mnt/workspace/zwq_data/training_llava_interleaved/checkpoints/test_debug \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --freeze_mm_mlp_adapter False \
    --h100 False \
    --mmc4_max_num_images 12