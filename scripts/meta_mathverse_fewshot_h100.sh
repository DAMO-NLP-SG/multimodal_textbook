#!/bin/bash

# 设置要使用的卡号
gpus=(0 1 2 3)

# 设置不同的 shots 数量
shots_list=(0 1 2 4)

# 模型路径
model_path="/mnt/model/llava-v1.5-7b-sft"
run_name=$(basename "$model_path")

prompt_mode="llava_v1"

including_self_in_first_example=true  # 或者根据需要设定为 false
random_select=false

if [ "$including_self_in_first_example" = true ]; then
    include='include'
else
    include='exclude'
fi

# 循环启动每个 shot 的任务
for i in "${!shots_list[@]}"; do
    shots=${shots_list[$i]}
    gpu=${gpus[$i]}

    # 设置输出文件名
    answers_file="/mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathverse/answers_zwq/${run_name}_${shots}shot_${include}.json"

    # 启动 Mathverse loader
    CUDA_VISIBLE_DEVICES="$gpu" python -m llava.eval.model_vqa_loader_few_shot_mathverse \
        --model-path ${model_path} \
        --question-file /mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathverse/testmini.json \
        --image-folder /mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathverse/images \
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
        --dataset_name mathverse \
        --random_select True \
        --including_self_in_first_example ${including_self_in_first_example} &

    # 等待一小会儿再启动下一个进程
    sleep 5
done

# 等待所有进程完成
wait

# 循环计算每个 shot 的评估结果
for shots in "${shots_list[@]}"; do
    answers_file="/mnt/data/zh/zwq/multimodal_textbook/playground/data/eval/mathverse/answers_zwq/${run_name}_${shots}shot_${include}.json"
    echo ${answers_file}
    python -m llava.eval.eval_mathverse \
        --result_file ${answers_file} \
        --backbone_llm deepseek_v2
done

echo "所有 shot 的评估结果已完成。"