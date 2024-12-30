import requests
from PIL import Image
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration
import torch

image1 = Image.open("/mnt/workspace/interleaved_LMM/test_case/1.png")
image2 = Image.open("/mnt/workspace/interleaved_LMM/test_case/2.png")
image3 = Image.open("/mnt/workspace/interleaved_LMM/test_case/3.png")

images = [image1, image2]

base_model_path = '/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b-base'
instruct_model_path = '/mnt/workspace/zwq_data/model/HuggingFaceM4/idefics2-8b'
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Can you describe these image in detail?"},
        {"type": "image"},
        {"type": "image"}
    ],
},
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "The first image is about a woman. The second image is about a dog."},
    ],
}]

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Idefics2Processor.from_pretrained(instruct_model_path, do_image_splitting=False)
model = Idefics2ForConditionalGeneration.from_pretrained(instruct_model_path,torch_dtype=torch.float16)
model.to(device)

text = processor.apply_chat_template(messages, add_generation_prompt=False)        
# 经过apply_chat_template，message 被整合成 'User: What’s the difference between these two images?<image><image><end_of_utterance>\nAssistant: The difference is that one image is about dogs and the other one about cats.<end_of_utterance>\n'

inputs = processor(images=images, text=text, return_tensors="pt").to(device)       
labels = inputs.input_ids.clone()
labels[labels == processor.tokenizer.pad_token_id] = -100   # pad_token_id对应的是 0
labels[labels == model.config.image_token_id] = -100        # image_token_id对应的是32001  fake_token_around_image对应的是 32000，每张图片变成<fake_token_around_image><image_token_id>...<image_token_id><fake_token_around_image>
# 对于每一张image 而言tokenize后的格式都是<fake_token_around_image> <image_token_id>*64 <fake_token_around_image>, 相当于第一张图 32000 32001....32001 32000 第二张图 32001....32001 32000 。end_of_utterance=32002
inputs["labels"] = labels

outputs = model(**inputs)
# loss = outputs.loss
# loss.backward()