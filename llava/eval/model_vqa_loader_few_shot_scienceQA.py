import argparse
import torch
import argparse
import torch
import os
import re
import json
import random
import warnings
import numpy as np
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.conversation_2 import conv_templates as conv_templates_2
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
from llava.eval.rices import RICES

class OpenflamingoVQADataset(Dataset):
    # For the training dataset, that is, the subsequent few-shot demonstration.
    def __init__(
        self, question_path, is_train, dataset_name
    ):
        with open(question_path, 'r') as file:
            self.dataset = json.load(file)
        self.is_train = is_train
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_path = item['image']

        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": item["text"],
            "question_id": item["question"],
        }
        
        answer_idx = item['answer']
        # mapped to ABCD 
        answer_mapping = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E'}
        mapped_answer = answer_mapping[str(answer_idx)]
        results["answers"] = mapped_answer
        
        return results



# Custom dataset class
class CustomFewShotDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, dataset_name, 
                 num_shots=None, rices=None, sampled_query_set=None):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.dataset_name = dataset_name
        # >>> few-shot init from open_flamingo >>>
        self.num_shots = num_shots
        self.rices = rices
        if self.rices:
            self.rices_dataset = sampled_query_set
        else:
            self.query_set = sampled_query_set

    def __getitem__(self, index):
        line = self.questions[index]
        if args.conv_mode in conv_templates:
            conv = conv_templates[args.conv_mode].copy()
        else:
            conv = conv_templates_2[args.conv_mode].copy()
        
        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file))   
        # >>> few-shot sampling from open_flamingo >>>
        if self.rices:
            batch_demo_samples = self.rices_dataset.find([image], self.num_shots)
        else:
            batch_demo_samples = open_flamingo.eval.utils.sample_batch_demos_from_query_set(
                self.query_set, self.num_shots, len([image]))
        # <<< few-shot sampling from open_flamingo <<<
        
        # >>> few-shot from open_flamingo combine with image processing of llava >>>
        context_images = [x["image"].convert('RGB') for x in batch_demo_samples[0]] if self.num_shots > 0 else []
        batch_images = context_images + [image]
        image_tensor = process_images(batch_images, self.image_processor, self.model_config).unsqueeze(1)
        # <<< few-shot from open_flamingo combine with image processing of llava <<<
        
        # >>> few-shot from open_flamingo combine with conversation of llava >>>
        for x in batch_demo_samples[0]:
            
            _context_q =  x["question"] + "Answer with the option\'s letter from the given choices directly."
            _context_a = x["answers"]
            conv.append_message(conv.roles[0], _context_q)
            conv.append_message(conv.roles[1], _context_a)
        # <<< few-shot from open_flamingo combine with conversation of llava <<<
        
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()



        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # if index <= 5:
        #     print(f"######## Sample {index} ########")
        #     print(prompt)
        #     print(f"######## Sample {index} ########")
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, image_tensor

    def __len__(self):
        return len(self.questions)




def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, dataset_name, batch_size=1, num_workers=4, num_shots=None, rices=None, sampled_query_set=None):
    assert batch_size == 1, "batch_size must be 1"
    assert num_workers == 0, "num_workers must be 0"
    dataset = CustomFewShotDataset(questions, image_folder, tokenizer, image_processor, model_config, dataset_name, num_shots=num_shots, rices=rices, sampled_query_set=sampled_query_set)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def split_list_v2(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    base_chunk_size = len(lst) // n  # integer division
    remainder = len(lst) % n  # remaining elements
    chunks = []
    for i in range(n):
        chunk_size = base_chunk_size + (i < remainder)  # add one to the chunk size for the first 'remainder' chunks
        start = i * base_chunk_size + min(i, remainder)  # calculate the start index
        chunks.append(lst[start:start+chunk_size])
    return chunks



def set_random_seed(seed, deterministic: bool = False) -> int:
    print(f"Setting seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print(
                'torch.backends.cudnn.benchmark is going to be set as '
                '`False` to cause cuDNN to deterministically select an '
                'algorithm')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # if digit_version() >= digit_version('1.10.0'):  # NOTE check manually
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return seed


def get_chunk(lst, n, k):
    chunks = split_list_v2(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    if args.distributed_launch:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


    if args.cached_demonstration_features is not None:
        cached_features = torch.load(
            f"{args.cached_demonstration_features}/{args.dataset_name}.pkl", map_location="cpu"
        )
    else:
        cached_features = None

    
    dataset_kwargs = dict(
        question_path=args.train_question_path,
        dataset_name=args.dataset_name,
        is_train=True,
    )

    VQADataset = OpenflamingoVQADataset

    train_dataset = VQADataset(**dataset_kwargs)
    print(f"{train_dataset.__class__.__name__} got")

    if args.rices:
        rices_dataset = RICES(
            train_dataset,
            device='cuda',
            batch_size=1,  # useless if cached_features is provided
            cached_features=cached_features,
            vision_encoder_path=args.rices_vision_encoder_path,
            vision_encoder_pretrained=args.rices_vision_encoder_pretrained,
        )
        sampled_query_set = rices_dataset
        if args.num_workers != 0:
            _may_be_err_msg = "RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method"
            warnings.warn(f"Set num_workers = 0, or you may raise:\n\t{_may_be_err_msg}")
            args.num_workers = 0
    else:
        query_set = open_flamingo.eval.utils.get_query_set(train_dataset, args.query_set_size)
        sampled_query_set = query_set
    print("sampled_query_set got")
    
    set_random_seed(args.seed, args.deterministic)
    
    # MODEL
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args.dataset_name, num_workers=args.num_workers, 
                                     num_shots=args.shots, rices=args.rices, sampled_query_set=sampled_query_set)

    print("dataloader got")


    # prepare generating kwargs
    if hasattr(args, "generate_kwargs"):
        generate_kwargs = args.generate_kwargs
    else:
        generate_kwargs = dict(
            do_sample=True if args.temperature > 0 else False,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generate_kwargs["top_p"] = args.top_p

    # NOTE use interleaved prepare
    model.use_openflamingo_interleaved_preparing = True
    model.disable_tokenizer_max_length = True
    model.disable_max_num_images = True
    
    if args.distributed_launch:
        torch.distributed.barrier()


    for (input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions), desc=answers_file):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        input_ids = input_ids.to(device=args.device, non_blocking=True)
        # image_file = line["image"]
        # image = Image.open(os.path.join(args.image_folder, image_file))
        # image_tensor = process_images([image], image_processor, model.config)[0]
        # images = image_tensor.unsqueeze(0).half().cuda()
        # image_sizes = [image.size]
        if getattr(model.config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        cur_prompt = '<image>' + '\n' + cur_prompt


        # if args.single_pred_prompt:
        #     qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
        #     cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        # if args.conv_mode in conv_templates:
        #     conv = conv_templates[args.conv_mode].copy()
        # else:
        #     conv = conv_templates_2[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        #input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device=args.device, non_blocking=True),
                **generate_kwargs)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--shots", default=0, type=int)
    parser.add_argument("--rices", type=bool, default=True)
    parser.add_argument("--query_set_size", type=int, default=2048, help="Size of demonstration query set")
    parser.add_argument("--rices_vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--rices_vision_encoder_pretrained", default="/mnt/workspace/zwq_data/model/openai/ViT-L-14.pt", type=str)
    parser.add_argument("--cached_demonstration_features", default=None)
    parser.add_argument("--train_image_dir_path", default=None, type=str)
    parser.add_argument("--train_question_path", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--deterministic", type=bool, default=False)
    parser.add_argument("--dataset_name", default="textvqa", type=str)
    parser.add_argument("--use-postprocess", type=bool, default=False)
    
    parser.add_argument("--distributed_launch", type=bool, default=False)
    

    args = parser.parse_args()
    print(args)

    if args.distributed_launch:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        args.num_chunks = world_size
        args.chunk_idx = rank
        folder = args.answers_file.replace(".jsonl", "")
        args.answers_file = f"{folder}/{args.num_chunks}_{args.chunk_idx}.jsonl"
        print(f"Rank {rank}/{world_size} is running chunk {args.chunk_idx}/{args.num_chunks}, "
              f"results will be saved to {args.answers_file}")
        # manage device
        device_id = rank % torch.cuda.device_count()
        args.device = f'cuda:{device_id}'
    else:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

        
    
    if args.model_base is None and "::" in args.model_path:
        model_base_and_path = args.model_path
        args.model_base, args.model_path = model_base_and_path.split("::")
        print(f"model_base_and_path ({model_base_and_path}) has been split into model_path ({args.model_path}) and model_base ({args.model_base})")

    eval_model(args)
