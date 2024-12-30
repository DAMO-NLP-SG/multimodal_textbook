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
from utils import short_answer, make_prompt, read_image,make_prompt_debug
from score_mathvista import score_mathvista
# Placeholder and system message
attn_implementation = "flash_attention_2"

    

# Evaluation results of mathvista

few_shot_reasoning_path = {
    864: 'Hot Pink clearly has lower values compared to Indian Red and Light Salmon. Specifically, the lowest value for Hot Pink is around 20, while the lowest values for the other two colors are above 40.',
    394: 'The image shows a mixture of adults and children in a park. The average age of the people is estimated to be 10 years old, due to the presence of many children.',
    776: 'To find the perimeter of rhombus LMPQ, we start with the given side length of 10. Since all sides of a rhombus are equal, each side of LMPQ is 10. The perimeter of a rhombus is the total length of all four sides. Therefore, the perimeter is 4 times the length of one side, which is 40.',
    911: 'Given that World War II ended in 1945, and all 4 people in the image were born after the end of World War II.',
    430: 'If all mayflies die, the dragonfly nymphs, which feed on mayfly nymphs, would have a significant food source removed. This reduction in their food supply would likely cause the dragonfly nymph population to decrease.',
    265: 'We can examine each type: Cuneate: Symmetrical, with a wedge-shaped base. Obtuse: Symmetrical, with a rounded tip. Cordate: Symmetrical, with a heart-shaped base. Truncate: Symmetrical, with a squared-off tip. Oblique: Asymmetrical, with unequal sides. Among these, the oblique leaf shape is the most uneven.',
    988: 'we can see that the object "glow" in the "slug" category is preferred by the most people, with a total of 9 people.',
    886: 'Given that angle ABC is 70 degrees, the central angle AOC, which is twice the inscribed angle ABC, is 140 degrees.'
}


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

    dataset = load_dataset(args.dataset_path, split='testmini')

    directory = os.path.dirname(args.answers_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(args.answers_file):
        with open(args.answers_file, 'w') as f:
            json.dump([], f)


    import random
    random.seed(0)
    # 
    indices = random.sample(range(len(dataset)), args.shot_number)
    # indices = list(few_shot_reasoning_path.keys())
    # reason_path = list(few_shot_reasoning_path.values())
    # 
    few_shot_data = [dataset[i] for i in indices]
    #few_shot_questions_answers = [{'question': item['query'][6:]+ '. Let us think step by step: ' + path, 'answer': item['answer'], 'image_id': item['image'][:-4]} for item, path in zip(few_shot_data,reason_path)]
    few_shot_questions_answers = [{'question': item['query'][6:], 'answer': item['answer'], 'image_id': item['image'][:-4]} for item in few_shot_data]

    shots = [make_prompt(qa['question'], image_token, qa['answer']) for qa in few_shot_questions_answers]   
    few_shot_images = [read_image(qa['image_id'], prefix ='', image_path=args.dataset_path) for qa in few_shot_questions_answers]

    few_shot_text = ''.join(shots)
    prefix = '<s>Instruction: provide an answer to the question. Use the image to answer.\n'
    #prefix = ''

    few_shot_text = prefix + few_shot_text

        
    output_list = []
    for idx, item in enumerate(dataset):
        image_list = []
        question_only  = item['question']
        choice = choice = item.get('choices', None)
        question = item['query'][6:]
        image_id = item['image'][:-4]
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
            "question": question_only,
            "choice": choice,
            "prompt": question, 
            "output_text": generated_texts,
            "pred_ans": answer_texts,
            "image_id": image_id,
            "answers_gt": answer_gt,
            "example_number": args.shot_number
        })
        print('question: ', question_only)
        print('GT answer: ', answer_gt)
        print('pred answer: ', answer_texts)
        #print('reasoning path: ', reasoning_path)
        print('the number of the generated_texts:', len(output_list))
        print(args.answers_file)
        print('\n')

        #
        with open(args.answers_file, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False)

    print("Done!")

    score_mathvista(args.answers_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="None")   #          
    #parser.add_argument("--model_path", type=str, default = "/mnt/workspace/zwq_data/training_output/llava2.1w_61w_ours_interleaved_2e7_epoch1_from_idefics_based_full_parameters_gbatch2048_context7168")                               #      /mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base
    parser.add_argument("--dataset_path", type=str, default = "/mnt/workspace/zwq_data/dataset_benchmark/MathVista")
    parser.add_argument("--answers_file", type=str, default="/mnt/workspace/zwq_data/interleaved_evaluation/debug/debug_mathvista.json")
    parser.add_argument("--temperature", type=float, default = 0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--shot_number", type=int, default=8)
    args = parser.parse_args()
    evaluate_engine(args)


