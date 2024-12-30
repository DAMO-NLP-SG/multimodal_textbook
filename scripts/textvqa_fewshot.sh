#!/bin/bash
shots=2  # Number of shots, for example
model_path="./LLaVA/checkpoints/ours_llava"
run_name=$(basename "$model_path")
echo "$run_name"
prompt_mode="llava_v1"

python -m llava.eval.model_vqa_loader_few_shot \
    --model-path  ${model_path}\
    --question-file /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa/llava_textvqa_val_v051_without_ocr.jsonl \
    --image-folder /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa/train_images \
    --answers-file /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa/answers_zwq/${run_name}_${shots}shot_textvqa_without_ocr.jsonl \
    --temperature 0 \
    --cached_demonstration_features /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa \
    --conv-mode ${prompt_mode} \
    --shots "$shots" \
    --query_set_size 2048 \
    --rices True \
    --seed 42 \
    --deterministic True \
    --train_image_dir_path /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa/train_images \
    --train_question_path /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa/TextVQA_0.5.1_train_process.json \
    --train_annotations_path /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa/TextVQA_0.5.1_train_anno.json \
    --dataset_name textvqa


python -m llava.eval.eval_textvqa \
    --annotation-file /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /mnt/workspace/multimodal_textbook/playground/data/eval/textvqa/answers_zwq/${run_name}_${shots}shot_textvqa_without_ocr.jsonl
