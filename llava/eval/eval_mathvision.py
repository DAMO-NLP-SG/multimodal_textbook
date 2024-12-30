from openai import OpenAI, AzureOpenAI, APIError
import json
import os
import re
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

GPT4V_ENDPOINT = ""
                 
GPT4V_KEY = ''
client = AzureOpenAI(
            azure_endpoint = GPT4V_ENDPOINT, 
            api_key=GPT4V_KEY,  
            api_version="2024-02-15-preview"
            )

key = ''

client_deepseek = OpenAI(api_key = key, base_url="https://api.deepseek.com")




def make_prompt(pred_ans,  labeld_ans,  choice=''):
    pred_ans, choice, labeld_ans = str(pred_ans), str(choice), str(labeld_ans)
    return f"Predicted Answer: {pred_ans}. Choice: {choice}. Labeled Answer: {labeld_ans}. Please output you judgement."

def parse_judgement(output):
    match = re.search(r"###judge: (True|False)", output, re.IGNORECASE)
    return match.group(1) if match else None


def call_llm_gpt(messages):
    
    response = client.chat.completions.create(
        model='gpt4o-mini',
        messages=messages,
        max_tokens=100,
        temperature = 0.2,
        )
    res = response.choices[0].message.content 
    return res


def call_llm_deepseek(messages):
    response = client_deepseek.chat.completions.create(
        model = "deepseek-chat",
        messages = messages,
        max_tokens = 100,
        temperature = 0.2,
        stream=False
    )

    res = response.choices[0].message.content
    return res

def call_local_qwen2(messages, tokenizer, llm, sampling_params):
    messages_list = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    
    outputs = llm.generate(messages_list, sampling_params)
    
    res = outputs[0].outputs[0].text
    
    return res

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--backbone_llm', type=str, default='gpt4o')
    return parser.parse_args()

def score_mathvision(res_path, backbone_llm):
    case1_input = "Predicted Answer: [0, 2]. Choice: ['[0, 2]', '[3, 2]', '[2, 4]', '[-3, 4]']. Labeled Answer: [0, 2]. Please output you judgement."
    case1_output = "###judge: True"


    case2_input = "Predicted Answer: D. Choice: ['[0, 2]', '[3, 2]', '[2, 4]', '[-3, 4]']. Labeled Answer: [-3, 4]. Please output you judgement."
    case2_output = "###judge: True"


    case3_input = "Predicted Answer: Yes. Choice: ['Yes', 'No']. Labeled Answer: No. Please output you judgement."
    case3_output = "###judge: False"

    case4_input = "Predicted Answer: 36°. Choice: None. Labeled Answer: 27°. Please output you judgement."
    case4_output = "###judge: False"


    case5_input = "Predicted Answer: B. Choice: ['steelheads would decrease.', 'stickleback fry would increase.', 'predatory insects would increase.', 'predatory insects would decrease.']. Labeled Answer: predatory insects would decrease.. Please output you judgement."
    case5_output = "###judge: False"


    case6_input = "query: Predicted Answer: quarter past. Choice: ['half', 'quarter', 'clock', 'quarter to', 'quarter past']. Labeled Answer: quarter. Please output you judgement."
    case6_output = "###judge: False"
    
    case7_input = "query: Predicted Answer: A. Choice: ['white five', 'white three', 'white four', 'white one', 'white two']. Labeled Answer: white one. Please output you judgement."
    case7_output = "###judge: False"



    ress = json.load(open(res_path, "r"))
    results = []
    correct_count = 0
    total_count = 0
    print(f'res_pth: {res_path}   backbone_llm: {backbone_llm}')

    if backbone_llm =='qwen2': 
        from vllm import LLM, SamplingParams
        max_model_len, tp_size = 8192, 4
        model_name= '/mnt/workspace/zwq_data/model/Qwen/Qwen2-72B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
        sampling_params = SamplingParams(temperature=0.1, top_p=0.8, repetition_penalty=1.05, max_tokens=200)


    for res in tqdm(ress):
        messages = [
            {"role": "system", "content": "You are a mathematics teacher. You need to help me determine whether the answer predicted by the model is correct. The mathematics questions may be multiple-choice questions or essay questions. You need to judge whether the predicted answer is correct based on the given labeled answers and options. Finally, you need to output ###Judge: True or ###Judge: False"},
            {"role": "user", "content": case1_input},
            {"role": "assistant", "content": case1_output},
            {"role": "user", "content": case2_input},
            {"role": "assistant", "content": case2_output},
            {"role": "user", "content": case3_input},
            {"role": "assistant", "content": case3_output},
            {"role": "user", "content": case4_input},
            {"role": "assistant", "content": case4_output},
            {"role": "user", "content": case5_input},
            {"role": "assistant", "content": case5_output},
            {"role": "user", "content": case6_input},
            {"role": "assistant", "content": case6_output},
            {"role": "user", "content": case7_input},
            {"role": "assistant", "content": case7_output}
        ]
        choice1,pred_case1,labeled_case1  = res['choice'], res['pred_ans'], res['answers_gt']
        query = make_prompt(pred_case1, labeled_case1,choice1)
        messages.append({"role": "user", "content": query})
        try: 
            if backbone_llm == 'gpt4o':
                output = call_llm_gpt(messages)
                #print('using gpt4o for scoring')
            elif backbone_llm == 'deepseek_v2':
                output = call_llm_deepseek(messages)
                #print('using deepseek for scoring')
                
            elif backbone_llm =='qwen2': 
                output = call_local_qwen2(messages,tokenizer ,llm, sampling_params)


            judgement = parse_judgement(output)
            is_correct = (judgement.lower() == 'true')
        except:
            output = 'Error'
            is_correct = False
        # print(res_path)
        # print('query: ', query)
        # print('resposne: ', output)
        # print('judge: ', is_correct)
        # print(len(results),'\n')
        
        res['llm_response_qwen'] = output
        res['judge_qwen'] = is_correct

        results.append(res)
        
        if is_correct:
            correct_count += 1
        total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f'Accuracy: {accuracy:.2%}')

    # Save the results to a file
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)




if __name__ == "__main__":
    args = get_args()
    if args.result_file is not None:
        score_mathvision(args.result_file, args.backbone_llm)