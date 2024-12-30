# Copyright (c) Facebook, Inc. and its affiliates.
import re
import json
from tqdm import tqdm
from utils import EvalAIAnswerProcessor


# vqav2 scoring
test_set = '/mnt/workspace/zwq_data/dataset_benchmark/VQAv2/v2_OpenEnded_mscoco_test2015_questions.json'


pred_file = '/mnt/workspace/zwq_data/interleaved_evaluation/VQAv2/VQAv2_pred_random_8_shot_ours_hf_prompt_gb2048.json'
output_file = pred_file[:-5] + '_summit.json'


with open(pred_file, 'r') as f:
    results = json.load(f)

result_question_ids = {}
for inst in results:
    result_question_ids[inst['question_id']] =  inst['pred_ans']


with open(test_set, 'r') as f:
    test_set = json.load(f)['questions']

answer_processor = EvalAIAnswerProcessor()

all_answers = []

for x in test_set:
    if x['question_id'] not in result_question_ids:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': ''
            })
    else:
        all_answers.append({
            'question_id': x['question_id'],
            'answer': answer_processor(result_question_ids[x['question_id']])
        }) 

# for x in results:
#     all_answers.append({
#         'question_id': x['question_id'],
#         'answer': answer_processor(x['pred_ans'])
#     })

with open(output_file, 'w') as f:
    json.dump(all_answers, f)