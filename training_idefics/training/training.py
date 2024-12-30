import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import sys
import transformers
from transformers import Trainer
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import numpy as np
import random
import traceback
import os
from transformers import Idefics2Processor
from peft import LoraConfig, get_peft_model
local_rank = None


# 训练 idefics2 用我们的 textbook interleaved dataset  数据格式是老格式 video-clip 格式

ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["WANDB_DISABLED"] = "true"
# # 设置可见的 cuda
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  
# 设置为当前路径

def rank0_print(*args):
    if local_rank in [0, -1]:
        print(*args)


class CustomTrainer(Trainer):
       def training_step(self, model, inputs):
           try:
               return super().training_step(model, inputs)
           except Exception as e:
               print("Error in batch:", inputs)
               raise e

def process_images(images):
    """
    将一个 batch 内的多个样本中的多个图像调整为相同的尺寸，并返回嵌套的 PIL.Image 对象列表。
    
    :param images: (list of list of PIL.Image) 输入图像列表，其中每个列表包含多张图像
    :return: 经过处理后的图像列表，仍然是嵌套的 PIL.Image 对象，形状保持不变
    """
    
    # 计算所有图像的中位尺寸
    all_heights = []
    all_widths = []
    

    for image_list in images:
        for img in image_list:
            if not isinstance(img, Image.Image):
                raise ValueError("Each item in the list must be a PIL.Image object")
            width, height = img.size
            all_heights.append(height)
            all_widths.append(width)
    
    # 计算中位数尺寸
    median_height = int(np.median(all_heights))
    median_width = int(np.median(all_widths))
    desired_size = (median_width, median_height)
    
    processed_batches = []
    for image_list in images:
        processed_images = []
        for img in image_list:
            # 调整图像尺寸
            img_resized = img.resize(desired_size)
            # 加入处理后的图像到列表中
            processed_images.append(img_resized)
        
        processed_batches.append(processed_images)
    
    return processed_batches


def ensure_first_sample_has_image(images, texts):
    # Check if the first sample already has images
    if not images[0]:
        # Find the first sample with images
        for i in range(1, len(images)):
            if images[i]:
                # Swap the first sample with this sample
                images[0], images[i] = images[i], images[0]
                texts[0], texts[i] = texts[i], texts[0]
                break
    return images, texts

def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    scratch_model_path: Optional[str] = field(default=None) 


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})

    is_multimodal: bool = False
    base_folder: Optional[str] = field(default=None)  # added by lixin4ever
    h100: bool = False
    add_ocr: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    embed_max_length: int = field(
        default=3072,
        metadata={
            "help":
            "Maximum input_embeds length. Embeds will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    train_scratch: bool = False 
    high_res: bool = False  
    

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # if i ==128690:
        #     print('jrer')
        # try:在 dataloader中调用__getitem__方法时，获取一个样本的数据，包括 input_ids 和 labels和 video path  128690
        conversations = self.list_data_dict[i]['conversations']
        token_num = self.list_data_dict[i]['token_num']
        processor = self.data_args.processor
        #fake_image_token = processor.fake_image_token.content                       #  '<fake_token_around_image>'  32000
        image_token = processor.image_token.content                                  #  '<image>'                    32001
        end_video_token  =  '\n'                                                     #  processor.end_of_utterance_token.content    
        bos_token = processor.tokenizer.bos_token                                    #  <s>    start of a conversation
        #pad_token_id = processor.tokenizer.pad_token_id  

        #assert token_num < processor.tokenizer.model_max_length

        text_one_instance = '' + bos_token
        image_files_list = []
        # conversations中包括多个video clip，每个 video clip 包括0，1 或多帧图片和一个 asr
        # image_str包含两侧的fake token和中间 64 个image token，用于代表每张图片
        for clip in conversations:
            current_prompt  = ''
            
            vid = clip['vid'] 
            if vid == "End of a Video" :
                current_prompt = current_prompt + end_video_token + bos_token
                continue
            # if vid == "excess the context length":
            #     text_one_instance = text_one_instance + 'excess the context length' + '\n'
            #    continue

            if clip.get('keyframe_ssim'):     
                image_paths = clip['keyframe_ssim']
                if len(image_paths)>0:
                    for image_file in image_paths:
                        if self.data_args.h100:
                            # /mnt/workspace/zwq_data/interleaved_dataset/dataset_images_interval_7 替换成 /mnt/data/zh/zwq/zwq_data/interleaved_dataset/dataset_images_interval_7
                            # /mnt/workspace/zwq_data/dataset_benchmark/llava_dataset/LLaVA-CC3M-Pretrain-595K/ 替换成/mnt/data/zh/zwq/zwq_data/dataset_benchmark/llava_dataset/LLaVA-CC3M-Pretrain-595K/
                            image_file = image_file.replace('/mnt/workspace/', '/mnt/data/zh/zwq/')
                            # /mnt/group_data2/zhiqiang/mm_datasets/mmc4-core-ff-images 替换成 /mnt/data/zh/zwq/zwq_data/dataset_benchmark/mmc4/mmc4_core_ff_images
                            image_file = image_file.replace('/mnt/group_data2/zhiqiang/mm_datasets/mmc4-core-ff-images', '/mnt/data/zh/zwq/zwq_data/dataset_benchmark/mmc4/mmc4_core_ff_images')

                        try:
                            image_files_list.append(Image.open(image_file))
                            current_prompt = current_prompt + 'Image:'+ image_token # 对于 our interleaved 数据 组织成如下: asr<image><image><image> asr<image><image> asr<image> asr<image><image>....

                            if vid == "image_text_pair":
                                current_prompt = current_prompt + end_video_token + bos_token     
                                # 对于构造的 llava pair_data组织成如下: caption<image>\n<s>caption<image>\n<s>caption<image>\n<s>caption<image>.... 
                        except:
                            
                            continue    

            if clip.get('ocr_qwen2_vl_72b'):
                ocr = '\nWe can see these words from the image by OCR: ' + clip.get('ocr_qwen2_vl_72b')
                #ocr = ' The word in the image: ' + clip.get('ocr_internvl_8b_deduplicates')

            else:
                ocr = ''

    
            if clip.get('refined_asr'):
                #asr = ' The answer: ' + clip.get('refined_asr')
                asr = '\n' + clip.get('refined_asr')
            elif clip.get('asr'):
                asr = '\n' + clip.get('asr')
                
            else:
                asr = ''

            
            ocr = ocr.replace(image_token, 'image')
            asr = asr.replace(image_token, 'image') 
            if self.data_args.add_ocr:
                current_prompt =  current_prompt + ocr +  asr
            else: 
                current_prompt =  current_prompt + asr



            # 将text中的<image> 替换为[image]
            #current_prompt = current_prompt.replace(image_token, '[image]')
            
            #image_num = len(image_paths)
            #text = asr 
            


            # 对于某些clip没有图片，只有asr，则:asr<image><image><image> asr<image><image> asr<image> asr<image><image> asr asr asr
            text_one_instance = text_one_instance + current_prompt
                
        # 检查text_one_instance中的image_token个数
        #image_num = text_one_instance.count(image_token)  #* processor.image_seq_len
        #assert image_num == len(image_files_list)


        if isinstance(i, int):
            data_dict = dict(input_ids=text_one_instance, images=image_files_list)
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.Idefics2Processor
    model: transformers.Idefics2ForConditionalGeneration
    data_args: DataArguments
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        texts = []
        images = []
        image_token = self.processor.image_token.content
        fake_image_token =  self.processor.fake_image_token.content
        image_token_id = self.processor.tokenizer.additional_special_tokens_ids[self.processor.tokenizer.additional_special_tokens.index(image_token)]
        fake_image_token_id = self.processor.tokenizer.additional_special_tokens_ids[self.processor.tokenizer.additional_special_tokens.index(fake_image_token)]
        # instances 是一个batch的样本, 包括images和text
        for instance in instances:  
            images.append(instance['images'])
            texts.append(instance['input_ids'])
            if len(instance['images']) == 0:
                pass
                #print('this instance contain 0 images')
        # images中的是多个list, 每个list 包括多张图片，每个图片是一个PIL.Image对象, 每张图片的尺寸是不一样的，需要变成相同的尺寸

        if not images or all(not sublist for sublist in images):
            # batch内每个样本都没有 image, 则在 text 结尾增加一张图，并 mask 掉
            #print('All sample in this batch contain 0 images')
            texts[0] = texts[0] + image_token

            dummy_image_path = os.path.join(self.data_args.base_folder, 'dataset/token_num_distribution.png')
            dummy_image = Image.open(dummy_image_path)#.convert('RGB')
            images[0].append(dummy_image)

            batch = self.processor(text=texts, images= images, padding='longest', truncation=True, max_length = self.tokenizer.model_max_length, return_tensors="pt")

            batch['pixel_attention_mask'].fill_(0)

            input_ids = batch['input_ids']   
            attention_mask = batch['attention_mask']   
            #attention_mask[input_ids==fake_image_token_id] = 0 
            attention_mask[input_ids==image_token_id] = 0 
            batch['attention_mask'] = attention_mask



        elif images and all(sublist for sublist in images): 
            # batch中每个样本都有 image
            #resize_images = process_images(images)

            batch = self.processor(text=texts, images=images, padding='longest', truncation=True, max_length = self.tokenizer.model_max_length, return_tensors="pt")
        else:
            # 第三种情况, batch 中某些样本没有 image, 则保证第一个样本里存在image即可, 因为/home/pai/lib/python3.11/site-packages/transformers/models/idefics2/image_processing_idefics2.py的images_list = make_list_of_images(images)函数
            #print('Some sample in this batch contain 0 images')
            images, texts = ensure_first_sample_has_image(images, texts)
            #resize_images = process_images(images)

            batch = self.processor(text=texts, images=images, padding='longest', truncation=True, max_length = self.tokenizer.model_max_length, return_tensors="pt")



        labels = batch["input_ids"].clone()
        #print('labels.shape:', labels.shape)
        # <image> <fake_token_around_image> <pad> 不计算 loss，  <end_of_utterance> 计算 loss
        
        # 在 model 的 loss 计算里 ignore_index被设置为image_token_id
        labels[labels == self.processor.tokenizer.pad_token_id] = -100# image_token_id   #
        #labels[labels == fake_image_token_id] = image_token_id # -100
        labels[labels == image_token_id] = -100#image_token_id # -100  
        batch["labels"] = labels
        # input_ids = batch["input_ids"]
        # attention_mask = batch["attention_mask"]
        # attention_mask[labels == -100] = 0                        # attention_mask中原来只有<pad>位置为0，现在加上<image>位置也为0, <fake_image_token>都应该不算 loss
        # batch["attention_mask"] = attention_mask
        # pixel_values = batch["pixel_values"]                      # pixel_values的维度是(batch, max_image_num, RGB, H, W ) 
        # pixel_attention_mask = batch["pixel_attention_mask"]

        #print('input_ids.shape, pixel_values.shape:', input_ids.shape, pixel_values.shape)
        #batch['original_texts'] = texts
        
        return batch




def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model ) -> Dict:
    """Make dataset and collator for xsupervised fine-tuning. dataloader 先调用dataset的__getitem__方法, 然后调用data collector的__call__方法将一个batch的样本拼接成一个batch."""
    train_dataset = LazySupervisedDataset(tokenizer = tokenizer,
                                data_path = data_args.data_path,
                                data_args = data_args)   
    # 构建dataset，包含了所有数据，__getitem__方法返回一个样本.dataloader会调用这个方法
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, processor = data_args.processor, model=model, data_args=data_args) # 构建data collector，用于将一个batch的样本拼接成一个batch
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])  #['o_proj', 'out_proj', 'up_proj', 'down_proj', 'q_proj', 'fc1', 'gate_proj', 'v_proj', 'k_proj', 'fc2']

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def train():
    global local_rank
    set_seed(42)


    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank


    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    config._attn_implementation = "flash_attention_2"

    if training_args.train_scratch:
        print("##########Training from connector scratch")
        model = transformers.Idefics2ForConditionalGeneration.from_pretrained(
            model_args.scratch_model_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
  
    else:
        print("##########Training from pretrained model")
        model = transformers.Idefics2ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )


    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)





    if training_args.lora_enable:
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    processor = Idefics2Processor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    processor.image_processor.do_image_splitting = False
    processor.tokenizer.model_max_length = training_args.model_max_length
    processor.tokenizer.padding_side = 'left'

    
    if training_args.high_res:
        processor.image_processor.size['longest_edge'] = 980
        processor.image_processor.size['shortest_edge'] = 336
    else:
        processor.image_processor.size['longest_edge'] = 560
        processor.image_processor.size['shortest_edge'] = 336    
    data_args.processor = processor 
    #data_args.is_multimodal = True
    print("Current model:", model)
    data_module = make_supervised_data_module(tokenizer = processor.tokenizer,
                                              data_args = data_args, model=model )   # 构建 dataset 和 data collector

    model.config.hidden_size = model.config.text_config.hidden_size


    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.text_model.requires_grad_(False)
        model.model.vision_model.requires_grad_(False)
        model.lm_head.requires_grad_(False)


    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"#######Trainable parameter#####: {name}")  
    trainer = Trainer(model = model,
                    tokenizer = processor.tokenizer,
                    args = training_args,
                    **data_module)
    #model = trainer.model
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True


    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)




if __name__ == "__main__":
    print('#################')
    train()


