#!/bin/bash

shots=2 # Number of shots, for example
prompt_mode="llava_v1"

model_path="/mnt/LLaVA/checkpoints/ours_llava"
run_name=$(basename "$model_path")
echo "$run_name"

including_self_in_first_example=false  # Whether to include itself in the few shot example. See our paper (analysis experiment) for more details

if [ "$including_self_in_first_example" = true ]; then
    include='include'
else
    include='exclude'
fi

python -m llava.eval.model_vqa_loader_few_shot_mathvista \
    --model-path ${model_path} \
    --question-file /mnt/workspace/multimodal_textbook/playground/data/eval/mathvista/test_mini.json \
    --image-folder /mnt/workspace/multimodal_textbook/playground/data/eval/mathvista/images \
    --answers-file /mnt/workspace/multimodal_textbook/playground/data/eval/mathvista/answers_zwq/${run_name}_${shots}shot_${include}.json \
    --cached_demonstration_features /mnt/workspace/multimodal_textbook/playground/data/eval/mathvision/mathvision.pkl \
    --temperature 0 \
    --conv-mode ${prompt_mode} \
    --shots "$shots" \
    --query_set_size 2048 \
    --rices True \
    --seed 42 \
    --deterministic True \
    --train_image_dir_path /mnt/workspace/multimodal_textbook/playground/data/eval/mathvision/images \
    --train_image_dir_path /mnt/workspace/multimodal_textbook/playground/data/eval/mathvision/test.json \
    --dataset_name mathvista \
    --including_self_in_first_example ${including_self_in_first_example} \
    --random_select False


python -m llava.eval.eval_mathvista \
    --result_file /mnt/workspace/multimodal_textbook/playground/data/eval/mathvista/answers_zwq/${run_name}_${shots}shot_${include}.json \
    --backbone_llm gpt4o
