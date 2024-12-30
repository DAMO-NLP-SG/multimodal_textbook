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

    questions_val = json.load(open(args.val_question_file, "r"))['questions']
    annotations_val =json.load(open(args.val_annotations_file, "r"))['annotations']
    assert len(questions_val) == len(annotations_val)


    questions_test = json.load(open(args.question_file, "r"))['questions']


    directory = os.path.dirname(args.answers_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(args.answers_file):
        with open(args.answers_file, 'w') as f:
            json.dump([], f)

    #image_token = processor.image_token.content  
    bos_token = processor.tokenizer.bos_token
    import random
    random.seed(0)
    # 生成随机索引
    indices = random.sample(range(len(questions_val)), args.shot_number)

    # 根据索引获取问题
    few_shot_questions = [questions_val[i] for i in indices]
    few_shot_answers = [annotations_val[i] for i in indices]
    few_shot_questions_answers = [{'question': q['question'], 'answer': a['answers'][0]['answer'], 'image_id': q['image_id']} for q, a in zip(few_shot_questions, few_shot_answers) if q['question_id'] == a['question_id']]


    shots = [make_prompt(qa['question'], image_token, qa['answer']) for qa in few_shot_questions_answers]
    few_shot_images = [read_image(q['image_id'], prefix='COCO_val2014_', image_path=args.val_image_folder) for q in few_shot_questions_answers]

    few_shot_text = ''.join(shots)
    prefix = f'{bos_token}Instruction: provide an answer to the question. Use the image to answer.\n'
    few_shot_text = prefix + few_shot_text

        
    output_list = []
    for idx, item in enumerate(questions_test):
        image_list = []
        question_id  = item['question_id']
        question = item['question']
        image_id = item['image_id']
        prompt = few_shot_text + make_prompt(question, image_token)
        image_list.extend(few_shot_images)
        image_list.append(read_image(image_id, prefix='COCO_test2015_', image_path = args.image_folder))

        inputs = processor(text=[prompt], images=[image_list], padding=False, return_tensors="pt")
        input_len = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, num_beams=args.num_beams)
        # 从generated_ids中提取输出的 ids
        generated_ids = generated_ids[:, input_len:]
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 分离输入和输出
        extracted_answer = generated_texts.split("\n")[0].strip()

        output_list.append({
            "question_id": question_id,
            "prompt": question, 
            "output_text": generated_texts,
            "pred_ans": extracted_answer,
            "image_id": image_id,
            "example_number": args.shot_number
        })
        print('question: ', question)
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
    parser.add_argument("--model_path", type=str, default="None")   #      /mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base
    #parser.add_argument("--model_path", type=str, default="/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base")   #      /mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base
    parser.add_argument("--val-question-file", type=str, default="/mnt/workspace/zwq_data/dataset_benchmark/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--val-annotations-file", type=str, default="/mnt/workspace/zwq_data/dataset_benchmark/VQAv2/v2_mscoco_val2014_annotations.json")
    parser.add_argument("--val-image-folder", type=str, default='/mnt/workspace/zwq_data/dataset_benchmark/VQAv2/val2014')
    parser.add_argument("--question-file", type=str, default="/mnt/workspace/zwq_data/dataset_benchmark/VQAv2/v2_OpenEnded_mscoco_test-dev2015_questions.json")
    parser.add_argument("--image-folder", type=str, default='/mnt/workspace/zwq_data/dataset_benchmark/VQAv2/test2015')
    parser.add_argument("--answers_file", type=str, default="/mnt/workspace/zwq_data/interleaved_evaluation/vqav2/VQAv2_pred_random_8_shot_ours_hf_prompt_gb2048.json")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--shot_number", type=int, default=8)
    args = parser.parse_args()
    evaluate_engine(args)




