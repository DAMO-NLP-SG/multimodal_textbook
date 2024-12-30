# Copyright (c) Facebook, Inc. and its affiliates.
import re
import json
from tqdm import tqdm  
import os
import torch
from PIL import Image
import argparse
from vqa_tools.vqa import VQA
from vqa_tools.vqa_eval import OKVQAEval
from utils import short_answer



#okvqa evaluation result scoring
def okvqa_results_processor(pred_file, output_dir, anno_files, ques_files):
    with open(pred_file, 'r') as f:
        results = json.load(f)


    save_result = []
    for res in results:
        save_result.append({
            "question_id": res["question_id"],
            "answer": short_answer(res['pred_ans'])
        })
    result_file = os.path.join(output_dir + "okvqa_answer.json")
    with open(result_file, "w") as f:
        json.dump(save_result, f)
    
    vqa = VQA(anno_files, ques_files)
    vqa_result = vqa.loadRes(
        resFile=result_file, quesFile=ques_files,
    )

    vqa_scorer = OKVQAEval(vqa, vqa_result, n=2)
    vqa_scorer.evaluate()
    print(f"OKVQA accuracy: {vqa_scorer.accuracy}")
    return vqa_scorer.accuracy



'''
# 主函数 
if __name__ == "__main__":
    pred_file = '/mnt/workspace/zwq_data/interleaved_evaluation/okvqa/okvqa_8_shot_ours_prompt_hf_our52w+2wllava_lr2e7_epoch2_gbatch2048_context3072.json' 
    output_dir = '/mnt/workspace/zwq_data/interleaved_evaluation/okvqa/'
    anno_files = '/mnt/workspace/zwq_data/dataset_benchmark/OKVQA/mscoco_val2014_annotations.json'
    ques_files = '/mnt/workspace/zwq_data/dataset_benchmark/OKVQA/dataset/OpenEnded_mscoco_val2014_questions.json'


    score = okvqa_results_processor(pred_file, output_dir, anno_files, ques_files)
'''