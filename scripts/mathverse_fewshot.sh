#!/bin/bash
shots=2  # Number of shots, for example
model_path="/mnt/LLaVA/checkpoints/mmc4_llava"
run_name=$(basename "$model_path")

prompt_mode="llava_v1"

including_self_in_first_example=false  # 或者根据需要设定为 false
random_select=false
if [ "$including_self_in_first_example" = true ]; then
    include='include'
else
    include='exclude'
fi

python -m llava.eval.model_vqa_loader_few_shot_mathverse \
    --model-path ${model_path} \
    --question-file /mnt/workspace/multimodal_textbook/playground/data/eval/mathverse/testmini.json \
    --image-folder /mnt/workspace/multimodal_textbook/playground/data/eval/mathverse/images \
    --answers-file /mnt/workspace/multimodal_textbook/playground/data/eval/mathverse/answers_zwq/${run_name}_${shots}shot_${include}.json \
    --cached_demonstration_features /mnt/workspace/multimodal_textbook/playground/data/eval/mathvision/mathvision.pkl \
    --temperature 0 \
    --conv-mode ${prompt_mode} \
    --shots "$shots" \
    --query_set_size 2048 \
    --rices True \
    --seed 42 \
    --deterministic True \
    --train_image_dir_path /mnt/workspace/multimodal_textbook/playground/data/eval/mathvision/images \
    --train_question_path /mnt/workspace/multimodal_textbook/playground/data/eval/mathvision/test.json \
    --dataset_name mathverse \
    --including_self_in_first_example ${including_self_in_first_example} \
    --random_select ${random_select}


python -m llava.eval.eval_mathverse \
    --result_file /mnt/workspace/multimodal_textbook/playground/data/eval/mathverse/answers_zwq/${run_name}_${shots}shot_${include}.json \
    --backbone_llm gpt4o
