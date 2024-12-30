import argparse
import torch
import os
import json
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from torch.utils.data import Dataset, DataLoader
from utils import make_prompt, read_image
from PIL import Image
import math
from score_TextVqa import TextVQAAccuracyEvaluator
attn_implementation = "flash_attention_2"


# Evaluate textvqa

def eval_model(args):
    # Model
    DEVICE =  'cuda'

    base_model_path = args.model_path
    processor = AutoProcessor.from_pretrained(base_model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    questions = json.load(open(args.question_file, "r"))

    directory = os.path.dirname(args.answers_file)

    # 
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(args.answers_file):
        with open(args.answers_file, 'w') as f:
            json.dump([], f)

    output_list = []

    image_token = processor.image_token.content  
    bos_token = processor.tokenizer.bos_token

    
    import random
    random.seed(0)
    few_shot_questions_answers = random.sample(questions, args.shot_number)
    
    #few_shot_questions_answers = [(q['question'], q['answers'][0]) for q in questions[:args.shot_number]]
    shots = [make_prompt(qa['question'], image_token, qa['answers'][0]) for qa in few_shot_questions_answers]
    few_shot_images = [read_image(q['image_id'], prefix = '', image_path = args.image_folder) for q in few_shot_questions_answers]
    few_shot_text = ''.join(shots)
    prefix = f'{bos_token}Instruction: provide an answer to the question. Use the image to answer.\n'
    few_shot_text = prefix + few_shot_text

    for instance in questions:

        image_list = []
        question = instance['question']

        prompt = few_shot_text + make_prompt(question, image_token)
        question_id = instance['question_id']
        answer_list = instance['answers']
        image_id = instance['image_id']
        image_list.extend(few_shot_images)
        image_list.append(read_image(image_id, prefix = '', image_path=args.image_folder)) 

        inputs = processor(text=[prompt], images=[image_list], padding=False, return_tensors="pt")
        input_len = inputs['input_ids'].shape[1]
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=args.temperature, num_beams=args.num_beams)
        # 从generated_ids中提取输出的 ids
        generated_ids = generated_ids[:, input_len:]
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # 分离输入和输出
        extracted_answer = generated_texts.split("\n")[0].strip()
        #extracted_answer = generated_texts.split("###")[0].strip()



        output_list.append({
            "question_id": question_id,
            "prompt": question, 
            "output_text": generated_texts,
            "pred_ans": extracted_answer,
            "image_id": image_id,
            "answers_gt": answer_list,
            "example_number": args.shot_number
        })
        print('question: ', question)
        print(args.answers_file)    
        print('labeled answer: ', answer_list)
        print('pred answer: ', extracted_answer)
        print('the number of the generated_texts:', len(output_list))
        print('\n')

        #存储结果
        with open(args.answers_file, 'w') as f:
            json.dump(output_list, f)

    print("Done!")
    print("The results are saved in", args.answers_file)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="None")   #          /mnt/workspace/zwq_data/interleaved_output_debug/checkpoint-8750
    #parser.add_argument("--model-path", type=str, default="/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base")
    parser.add_argument("--question-file", type=str, default="/mnt/workspace/zwq_data/dataset_benchmark/TextVQA/TextVQA_0.5.1_val.json")
    parser.add_argument("--answers_file", type=str, default="None")
    parser.add_argument("--image-folder", type=str, default='/mnt/workspace/zwq_data/dataset_benchmark/TextVQA/train_images')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--shot_number", type=int, default=8)
    args = parser.parse_args()

    eval_model(args)


    evaluator = TextVQAAccuracyEvaluator()
    with open(args.answers_file, 'r') as f:
        pred_list = json.load(f)

    accuracy = evaluator.eval_pred_list(pred_list, output_path = args.answers_file)

    print(f"Accuracy: {accuracy}")