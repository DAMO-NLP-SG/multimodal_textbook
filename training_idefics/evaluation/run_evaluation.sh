#!/bin/bash
# pip uninstall transformers
# pip install transformers==4.40

# 设置变量
DATASET='okvqa'  # okvqa, textvqa, mathverse, mathvista, mathvision, vqav2
CHECKPOINT_NUM='320'

EVAL_SCRIPT="/mnt/workspace/interleaved_LMM/evaluation/eval_${DATASET}.py"

#MODEL_PATH="/mnt/workspace/zwq_data/training_output/$MODEL_NAME/checkpoint-${CHECKPOINT_NUM}"
MODEL_PATH="/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base"
MODEL_NAME="idefics2-8b-base"



# 
cp "/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base/preprocessor_config.json" "$MODEL_PATH"
cp "/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base/processor_config.json" "$MODEL_PATH"

set -e

# Iterate through different shot_numbers.
for shot_number in 0 1 2 4 8; do
  ANSWERS_FILE="/mnt/workspace/zwq_data/interleaved_evaluation/$DATASET/${DATASET}_${shot_number}_shot_ours_prompt_hf_${MODEL_NAME}.json"
  
  # 
  echo "Processing shot_number: $shot_number"
  echo $ANSWERS_FILE
  
  # 
  python "${EVAL_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --answers_file "${ANSWERS_FILE}" \
    --shot_number "${shot_number}"
    
  echo "Completed shot_number: $shot_number"
  echo $MODEL_PATH
done
