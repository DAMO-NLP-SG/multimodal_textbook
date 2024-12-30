# Copyright (c) Facebook, Inc. and its affiliates.
import re
import json
from tqdm import tqdm  
import os
import torch
from PIL import Image
import argparse
from llava.eval.vqa_tools.vqa import VQA
from llava.eval.vqa_tools.vqa_eval import OKVQAEval
from llava.eval.utils import short_answer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='')
    parser.add_argument('--pred_file', type=str, default =None)
    parser.add_argument('--question_file', type=str, default ='')
    parser.add_argument('--output_dir', type=str, default ='')
    return parser.parse_args()


def okvqa_results_processor(pred_file, output_dir, anno_files, ques_files):
    # 读取 jsonl 文件
    results = []
    with open(pred_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            results.append(json_obj)
    


    save_result = []
    for res in results:
        save_result.append({
            "question_id": int(res["question_id"]),
            "answer": short_answer(res['text'])
        })
    result_file = os.path.join(output_dir, "okvqa_answer.json")
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




# 主函数 
if __name__ == "__main__":


    args = get_args()
    if args.pred_file is not None:
        okvqa_results_processor(args.pred_file, args.output_dir, args.annotation_file, args.question_file)

