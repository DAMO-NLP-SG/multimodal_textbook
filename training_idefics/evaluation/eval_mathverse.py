import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from dataclasses import dataclass, field
import transformers
import argparse
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from vqa_tools.vqa import VQA
from vqa_tools.vqa_eval import OKVQAEval
from utils import short_answer, make_prompt, read_image
import re

from score_mathverse import score_mathverse

# Placeholder and system message

# Evaluation results of mathverse
attn_implementation = "flash_attention_2"

def evaluate_engine(args):
    device = 'cuda'
    base_model_path = args.model_path
    processor = AutoProcessor.from_pretrained(base_model_path)
    # processor.image_processor.size['longest_edge'] = 560
    # processor.image_processor.size['shortest_edge'] = 336    
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation = attn_implementation,
    )    


    image_token = processor.image_token.content  
    json_file = os.path.join(args.dataset_path, 'testmini.json')    
    with open(json_file, 'r') as f:
        dataset = json.load(f)
        


    directory = os.path.dirname(args.answers_file)

    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(args.answers_file):
        with open(args.answers_file, 'w') as f:
            json.dump([], f)

    import random
    random.seed(0)
    indices = random.sample(range(len(dataset)), args.shot_number)
    few_shot_data = [dataset[i] for i in indices]
    few_shot_questions_answers = [{'question': item['question_for_eval'], 'answer': item['answer'], 'image_id': item['image']} for item in few_shot_data]

    shots = [make_prompt(qa['question'], image_token, qa['answer']) for qa in few_shot_questions_answers]   
    few_shot_images = [read_image(os.path.join('images',qa['image_id']), prefix ='', image_path=args.dataset_path) for qa in few_shot_questions_answers]

    few_shot_text = ''.join(shots)
    prefix = '<s>Instruction: provide an answer to the question. Use the image to answer.\n'
    few_shot_text = prefix + few_shot_text

        
    output_list = []


    for idx, item in enumerate(dataset):
        image_list = []
        question  = item['question_for_eval']
        image_id = os.path.join('images', item['image'])
        answer_gt = item['answer']

        prompt = few_shot_text + make_prompt(question, image_token, Reasoning=False)
        image_list.extend(few_shot_images)
        image_list.append(read_image(image_id, prefix='', image_path=args.dataset_path)) 

        inputs = processor(text=[prompt], images=[image_list], padding=False, return_tensors="pt")
        input_len = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, num_beams=args.num_beams)
        # 从generated_ids中提取输出的 ids
        generated_ids = generated_ids[:, input_len:]
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 分离输入和输出
        
        answer_texts = generated_texts.split("\n")[0].strip()
        #answer_texts = generated_texts.split("###")[0].strip()

        #print('####',extracted_answer )
        # try: 
        #     reasoning_path, answer_texts = extracted_answer.split("Answer:")
        #     answer_texts = answer_texts.strip()
        # except:
        #     #reasoning_path = ''
        #     answer_texts = 'Error'


        output_list.append({
            "question": question,
            "prompt": question, 
            "output_text": generated_texts,
            "pred_ans": answer_texts,
            "image_id": image_id,
            "answers_gt": answer_gt,
            "example_number": args.shot_number
        })
        print('GT answer: ', answer_gt)
        print('pred answer: ', answer_texts)
        #print('reasoning path: ', reasoning_path)
        print('the number of the generated_texts:', len(output_list))
        print(args.answers_file)
        print('\n')

        #存储结果
        with open(args.answers_file, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False)

    print("Done!")


    score_mathverse(args.answers_file)   




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="None")   #          
    #parser.add_argument("--model_path", type=str, default = "/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base")            #      /mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base
    parser.add_argument("--dataset_path", type=str, default = "/mnt/workspace/zwq_data/dataset_benchmark/MathVerse")
    parser.add_argument("--answers_file", type=str, default="/mnt/workspace/zwq_data/interleaved_evaluation/MathVerse/mathverse_pred_random_8_shot_ours_hf_prompt_hf_our52w+2wllava_lr2e7_epoch2_gbatch2048_context3072.json")
    parser.add_argument("--temperature", type=float, default = 0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--shot_number", type=int, default=8)
    args = parser.parse_args()
    evaluate_engine(args)