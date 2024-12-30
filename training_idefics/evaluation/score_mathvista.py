from openai import OpenAI
import json
import os
import re
key = 'sk-'

client = OpenAI(api_key = key, base_url="https://api.deepseek.com")


# MathVista evaluation result scoring

def make_prompt(pred_ans,  labeld_ans,  choice=''):
    pred_ans, choice, labeld_ans = str(pred_ans), str(choice), str(labeld_ans)
    return f"Predicted Answer: {pred_ans}. Choice: {choice}. Labeled Answer: {labeld_ans}. Please output you judgement."

def parse_judgement(output):
    match = re.search(r"###judge: (True|False)", output, re.IGNORECASE)
    return match.group(1) if match else None


def call_llm(messages):
    response = client.chat.completions.create(
        model = "deepseek-chat",
        messages = messages,
        max_tokens = 64,
        temperature = 0.2,
        stream=False
    )

    res = response.choices[0].message.content
    return res

def score_mathvista(res_path):
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



    ress = json.load(open(res_path, "r"))
    results = []
    correct_count = 0
    total_count = 0

    for res in ress:
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
            {"role": "assistant", "content": case6_output}
        ]
        choice1,pred_case1,labeled_case1  = res['choice'], res['pred_ans'], res['answers_gt']
        query = make_prompt(pred_case1, labeled_case1,choice1)
        messages.append({"role": "user", "content": query})
        try: 
            output = call_llm(messages)
            judgement = parse_judgement(output)
            is_correct = (judgement.lower() == 'true')
        except:
            output = 'Error'
            is_correct = False
        print(res_path)
        print('query: ', query)
        print('resposne: ', output)
        print('judge: ', is_correct)
        print(len(results),'\n')
        
        res['llm_response'] = output
        res['judge'] = is_correct

        results.append(res)
        
        if is_correct:
            correct_count += 1
        total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f'Accuracy: {accuracy:.2%}')

    # Save the results to a file
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)



