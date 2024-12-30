#!/bin/bash

# Set the card number to be used.
gpus=(0 1 2 3)

# Set different numbers of shots.
shots_list=(0 1 2 4)

# 
model_path="/mnt/model/llava-v1.5-7b-sft"
run_name=$(basename "$model_path")

prompt_mode="llava_v1"

including_self_in_first_example=false  #

if [ "$including_self_in_first_example" = true ]; then
    include='include'
else
    include='exclude'
fi

for i in "${!shots_list[@]}"; do
    shots=${shots_list[$i]}
    gpu=${gpus[$i]}


    answers_file="/mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathvista/answers_zwq/${run_name}_${shots}shot_${include}.json"


    CUDA_VISIBLE_DEVICES="$gpu" python -m llava.eval.model_vqa_loader_few_shot_mathvista \
        --model-path ${model_path} \
        --question-file /mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathvista/test_mini.json \
        --image-folder /mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathvista/images \
        --answers-file ${answers_file} \
        --cached_demonstration_features /mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathvision/mathvision.pkl \
        --temperature 0 \
        --conv-mode ${prompt_mode} \
        --shots "$shots" \
        --query_set_size 2048 \
        --rices True \
        --seed 42 \
        --deterministic True \
        --train_image_dir_path /mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathvision/images \
        --train_question_path /mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathvision/test.json \
        --dataset_name mathvista \
        --including_self_in_first_example ${including_self_in_first_example} &


    sleep 5
done

wait


for shots in "${shots_list[@]}"; do
    answers_file="/mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathvista/answers_zwq/${run_name}_${shots}shot_${include}.json"
    
    echo ${answers_file}
    python -m llava.eval.eval_mathvista \
        --result_file ${answers_file} \
        --backbone_llm deepseek_v2
done

echo "all shots evaluation done"