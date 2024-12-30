import os
import cv2
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import imageio
import torch
import sys
import transformers
from transformers import Trainer
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModel,Idefics2Processor

import numpy as np
import random
import traceback
import os
from peft import LoraConfig, get_peft_model
local_rank = None
import torch.nn.functional as F
import math
def interpolate_pos_embedding(patch_pos_embed, num_h_patches, num_w_patches):
    # 计算 position embedding 的原始尺寸
    num_positions = patch_pos_embed.shape[0]  # 729
    sqrt_num_positions = int(math.sqrt(num_positions)) #27

    # Reshape to (1, sqrt_num_positions, sqrt_num_positions, C) and permute to (1, C, sqrt_num_positions, sqrt_num_positions)
    patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, -1).permute(0, 3, 1, 2)

    # 检查并处理数据类型
    fp32_upcasting = patch_pos_embed.dtype == torch.bfloat16
    if fp32_upcasting:
        patch_pos_embed = patch_pos_embed.to(torch.float32)

    # 计算 scale_factor
    scale_factor = (num_h_patches / sqrt_num_positions, num_w_patches / sqrt_num_positions)  # (2.59, 2.59)
    print(scale_factor)
    # 使用 scale_factor 进行插值
    patch_pos_embed = F.interpolate(
        patch_pos_embed,
        scale_factor=scale_factor,
        mode="bicubic",
        align_corners=False
    )

    # 如果之前进行了数据类型转换，重新转回 bfloat16
    if fp32_upcasting:
        patch_pos_embed = patch_pos_embed.to(torch.bfloat16)

    # 再次 permute 并 reshape 回原始形状
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(num_h_patches * num_w_patches, -1)

    return patch_pos_embed




def load_mistral_weights_to_idefics2(idefics2_model, mistral_model):
    # 加载embed_tokens
    idefics2_model.model.text_model.embed_tokens.weight.data[:32000, :] = mistral_model.model.embed_tokens.weight.data
    
    # 加载layers
    for i in range(32):
        # Self Attention
        idefics2_model.model.text_model.layers[i].self_attn.q_proj.weight.data = mistral_model.model.layers[i].self_attn.q_proj.weight.data
        idefics2_model.model.text_model.layers[i].self_attn.k_proj.weight.data = mistral_model.model.layers[i].self_attn.k_proj.weight.data
        idefics2_model.model.text_model.layers[i].self_attn.v_proj.weight.data = mistral_model.model.layers[i].self_attn.v_proj.weight.data
        idefics2_model.model.text_model.layers[i].self_attn.o_proj.weight.data = mistral_model.model.layers[i].self_attn.o_proj.weight.data
        
        # MLP
        idefics2_model.model.text_model.layers[i].mlp.gate_proj.weight.data = mistral_model.model.layers[i].mlp.gate_proj.weight.data
        idefics2_model.model.text_model.layers[i].mlp.up_proj.weight.data = mistral_model.model.layers[i].mlp.up_proj.weight.data
        idefics2_model.model.text_model.layers[i].mlp.down_proj.weight.data = mistral_model.model.layers[i].mlp.down_proj.weight.data
        
        # LayerNorms
        idefics2_model.model.text_model.layers[i].input_layernorm.weight.data = mistral_model.model.layers[i].input_layernorm.weight.data
        idefics2_model.model.text_model.layers[i].post_attention_layernorm.weight.data = mistral_model.model.layers[i].post_attention_layernorm.weight.data
    
    # 加载最终的norm
    idefics2_model.model.text_model.norm.weight.data = mistral_model.model.norm.weight.data
    
    # 加载lm_head
    idefics2_model.lm_head.weight.data[:32000, :] = mistral_model.lm_head.weight.data


def load_siglip_weights_to_idefics2(idefics2_model, siglip_model):
    idefics2_model.model.vision_model.embeddings.patch_embedding.weight.data = siglip_model.vision_model.embeddings.patch_embedding.weight.data
    
    pos_emd = siglip_model.vision_model.embeddings.position_embedding.weight.data
    pos_emd_ = interpolate_pos_embedding(pos_emd, 70, 70)
    idefics2_model.model.vision_model.embeddings.position_embedding.weight.data = pos_emd_   


    for i in range(27):
        idefics2_model.model.vision_model.encoder.layers[i].self_attn.q_proj.weight.data = siglip_model.vision_model.encoder.layers[i].self_attn.q_proj.weight.data
        idefics2_model.model.vision_model.encoder.layers[i].self_attn.k_proj.weight.data = siglip_model.vision_model.encoder.layers[i].self_attn.k_proj.weight.data
        idefics2_model.model.vision_model.encoder.layers[i].self_attn.v_proj.weight.data = siglip_model.vision_model.encoder.layers[i].self_attn.v_proj.weight.data
        idefics2_model.model.vision_model.encoder.layers[i].self_attn.out_proj.weight.data = siglip_model.vision_model.encoder.layers[i].self_attn.out_proj.weight.data

        idefics2_model.model.vision_model.encoder.layers[i].layer_norm1.weight.data = siglip_model.vision_model.encoder.layers[i].layer_norm1.weight.data


        idefics2_model.model.vision_model.encoder.layers[i].mlp.fc1.weight.data = siglip_model.vision_model.encoder.layers[i].mlp.fc1.weight.data
        idefics2_model.model.vision_model.encoder.layers[i].mlp.fc2.weight.data = siglip_model.vision_model.encoder.layers[i].mlp.fc2.weight.data

        idefics2_model.model.vision_model.encoder.layers[i].layer_norm2.weight.data = siglip_model.vision_model.encoder.layers[i].layer_norm2.weight.data

    idefics2_model.model.vision_model.post_layernorm.weight.data = siglip_model.vision_model.post_layernorm.weight.data



def train():
    global local_rank

    mistral_model = AutoModelForCausalLM.from_pretrained("/mnt/workspace/zwq_data/model/models--mistralai--Mistral-7B-v0.1/snapshots/f9824c3c1090baa2f7c1d33ddc89becab37b3e18")
    siglip_model = AutoModel.from_pretrained("/mnt/workspace/zwq_data/model/siglip-so400m")


    config = transformers.AutoConfig.from_pretrained('/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base', trust_remote_code=True)

    config._attn_implementation = "flash_attention_2"

    
    model = transformers.Idefics2ForConditionalGeneration(config)
    load_siglip_weights_to_idefics2(model, siglip_model)
    load_mistral_weights_to_idefics2(model, mistral_model)
    
    #  保存 model 参数
    output_dir = "/mnt/workspace/zwq_data/model/idefics2-8b-from-mistral-siglip"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model.save_pretrained(output_dir)



  



if __name__ == "__main__":
    train()

