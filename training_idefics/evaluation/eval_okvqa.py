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

from vqa_tools.vqa import VQA
from vqa_tools.vqa_eval import OKVQAEval
from utils import short_answer, make_prompt, read_image
from score_OKVQA import okvqa_results_processor

# Placeholder and system message
attn_implementation = "flash_attention_2"

    

# Evaluate okvqa

def evaluate_engine(args):

    device = 'cuda'
    base_model_path = args.model_path
    processor = AutoProcessor.from_pretrained(base_model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation = attn_implementation,
    )    

    image_token = processor.image_token.content  

    questions = json.load(open(args.question_file, "r"))
    annotations =json.load(open(args.annotations_file, "r"))['annotations']
    assert len(questions) == len(annotations)

    directory = os.path.dirname(args.answers_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(args.answers_file):
        with open(args.answers_file, 'w') as f:
            json.dump([], f)

    image_token = processor.image_token.content  

    import random
    random.seed(0)
    # 
    indices = random.sample(range(len(questions)), args.shot_number)

    # 
    few_shot_questions = [questions[i] for i in indices]
    few_shot_answers = [annotations[i] for i in indices]
    few_shot_questions_answers = [{'question': q['question'], 'answer': a['answers'][0]['answer'], 'image_id': q['image_id']} for q, a in zip(few_shot_questions, few_shot_answers) if q['question_id'] == a['question_id']]


    shots = [make_prompt(qa['question'], image_token, qa['answer']) for qa in few_shot_questions_answers]
    few_shot_images = [read_image(q['image_id'], prefix='COCO_val2014_', image_path=args.image_folder) for q in few_shot_questions_answers]
    few_shot_text = ''.join(shots)
    prefix = '<s>Instruction: provide an answer to the question. Use the image to answer.\n'
    #prefix = f'{bos_token}Instruction: provide an answer to the question. Use the image to answer.\n'


    few_shot_text = prefix + few_shot_text

        
    output_list = []
    for idx, item in enumerate(questions):
        image_list = []
        question_id  = item['question_id']
        question = item['question']
        image_id = item['image_id']
        answer_list = annotations[idx]['answers']
        prompt = few_shot_text + make_prompt(question, image_token)
        image_list.extend(few_shot_images)
        image_list.append(read_image(image_id, prefix='COCO_val2014_', image_path=args.image_folder)) 

        inputs = processor(text=[prompt], images=[image_list], padding=False, return_tensors="pt")
        input_len = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, num_beams=args.num_beams)
        #  
        generated_ids = generated_ids[:, input_len:]
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 
        extracted_answer = generated_texts.split("\n")[0].strip()
        #extracted_answer = extracted_answer.split("Question")[0].strip()
        #extracted_answer = generated_texts.strip()
        #extracted_answer = generated_texts.split("###")[0].strip()

        answers_list_gt = [ans['answer'] for ans in answer_list]

        output_list.append({
            "question_id": question_id,
            "prompt": question, 
            "output_text": generated_texts,
            "pred_ans": extracted_answer,
            "image_id": image_id,
            "answers_gt": answer_list,
            "answers_list_gt": answers_list_gt,
            "example_number": args.shot_number
        })
        print('question: ', question)
        print(args.answers_file)
        print('labeled answer: ', answers_list_gt)
        print('pred answer: ', extracted_answer)
        print('the number of the generated_texts:', len(output_list))
        print(args.answers_file)

        print('\n')

        #存储结果
        with open(args.answers_file, 'w') as f:
            json.dump(output_list, f)

    print("Done!")



    # # process results
    # metric = None
    # if dist.get_rank() == 0:
    #     metric = okvqa_results_processor(results, eval_args.output_path, **results_process_kwargs)
    
    # return metric

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/workspace/zwq_data/training_output/our52w_lr2e6_epoch2_gbatch2048_context3072_0830/checkpoint-200")   #          
    #parser.add_argument("--model_path", type=str, default="/mnt/workspace/zwq_data/training_output/595kllava_lr1e5_epoch1_gbatch1024_context1024_1node/checkpoint-100")   #          
    #parser.add_argument("--model_path", type=str, default="/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base")        #      /mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base
    parser.add_argument("--question_file", type=str, default="/mnt/workspace/zwq_data/dataset_benchmark/OKVQA/OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--annotations_file", type=str, default="/mnt/workspace/zwq_data/dataset_benchmark/OKVQA/mscoco_val2014_annotations.json")
    parser.add_argument("--image_folder", type=str, default='/mnt/workspace/zwq_data/dataset_benchmark/OKVQA/image_val2014')
    parser.add_argument("--answers_file", type=str, default="None")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--shot_number", type=int, default=8)
    args = parser.parse_args()
    evaluate_engine(args)




    okvqa_results_processor(args.answers_file, 
    '/mnt/workspace/zwq_data/interleaved_evaluation/okvqa/', 
    args.annotations_file, 
    '/mnt/workspace/zwq_data/dataset_benchmark/OKVQA/dataset/OpenEnded_mscoco_val2014_questions.json')

