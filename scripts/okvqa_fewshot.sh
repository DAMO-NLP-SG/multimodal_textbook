#!/bin/bash
model_path="/mnt/LLaVA/checkpoints/mmc4_llava"
run_name=$(basename "$model_path")
echo "$run_name"

shots=0 # Number of shot
conv_mode="llava_v1"
okvqa_root="/mnt/workspace/multimodal_textbook/playground/data/eval/okvqa"
answers_file="/mnt/workspace/multimodal_textbook/playground/data/eval/okvqa/answers_zwq/${run_name}_${shots}shots_prompt_${conv_mode}.jsonl"

#Run the evaluation command
python -m llava.eval.model_vqa_loader_few_shot \
  --model-path "$model_path" \
  --question-file "${okvqa_root}/OpenEnded_mscoco_val2014_questions.jsonl" \
  --image-folder "/mnt/workspace/zwq_data/dataset_benchmark/OKVQA/image_val2014" \
  --answers-file "$answers_file" \
  --temperature "0" \
  --conv-mode "$conv_mode" \
  --shots "$shots" \
  --query_set_size "2048" \
  --rices "True" \
  --seed "42" \
  --deterministic "True" \
  --cached_demonstration_features "/mnt/workspace/zwq_data/dataset_benchmark/OKVQA" \
  --train_image_dir_path "/mnt/workspace/zwq_data/dataset_benchmark/coco2014/train2014" \
  --train_question_path "${okvqa_root}/OpenEnded_mscoco_train2014_questions.json" \
  --train_annotations_path "${okvqa_root}/mscoco_train2014_annotations.json" \
  --dataset_name "ok_vqa"

# Run the OKVQA evaluation command
python -m llava.eval.eval_okvqa \
  --pred_file "$answers_file"
