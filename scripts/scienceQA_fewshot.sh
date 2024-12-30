#!/bin/bash

shots=2  # Number of shots, for example
model_path="./LLaVA/checkpoints/ours_llava"
run_name=$(basename "$model_path")
prompt_mode="llava_v1"
echo "$run_name"

python -m llava.eval.model_vqa_loader_few_shot_scienceQA \
    --model-path ${model_path} \
    --question-file /mnt/workspace/multimodal_textbook/playground/data/eval/scienceqa/filtered_llava_test_CQM-A.json \
    --image-folder /mnt/group_data/zwq_data/LLaVA/playground/data/eval/scienceqa/images/test \
    --answers-file /mnt/workspace/multimodal_textbook/playground/data/eval/scienceqa/answers_zwq/${run_name}_${shots}shot.jsonl \
    --cached_demonstration_features /mnt/workspace/multimodal_textbook/playground/data/eval/scienceqa \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode ${prompt_mode} \
    --shots "$shots" \
    --query_set_size 2048 \
    --rices True \
    --seed 42 \
    --deterministic True \
    --train_image_dir_path /mnt/group_data/zwq_data/LLaVA/playground/data/eval/scienceqa/images/train \
    --train_question_path /mnt/workspace/multimodal_textbook/playground/data/eval/scienceqa/train_with_image.json \
    --dataset_name scienceQA

python /mnt/workspace/multimodal_textbook/llava/eval/eval_science_qa.py \
    --base-dir /mnt/workspace/multimodal_textbook/playground/data/eval/scienceqa \
    --result-file /mnt/workspace/multimodal_textbook/playground/data/eval/scienceqa/answers_zwq/${run_name}_${shots}shot.jsonl \
    --output-file /mnt/workspace/multimodal_textbook/playground/data/eval/scienceqa/answers_zwq/${run_name}_${shots}shot_scoreres.jsonl \
    --output-result /mnt/workspace/multimodal_textbook/playground/data/eval/scienceqa/answers_zwq/${run_name}_${shots}shot_sqares.json
