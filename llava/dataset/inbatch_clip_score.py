import json
import argparse
import cv2
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
import itertools

# 加载 CLIP 模型
# 检查是否有可用的 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("/mnt/data/zwq_data/model/openai/clip-vit-large-patch14-336").to(device)
processor = CLIPProcessor.from_pretrained("/mnt/data/zwq_data/model/openai/clip-vit-large-patch14-336")


def resize_image(image, target_size):
    # 将图片 resize 成指定大小
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def compare_images(image1_path, image2_path):
    try:
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if image1 is None or image2 is None:
            print(f"Error reading images: {image1_path}, {image2_path}")
            return None  # 返回 None 表示处理失败

        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        inputs = processor(images=[image1, image2], return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)

        image1_features = image_features[0]
        image2_features = image_features[1]
        clip_score = torch.nn.functional.cosine_similarity(image1_features, image2_features, dim=0).item()

        return clip_score
    except Exception as e:
        print(f"Error processing images {image1_path} and {image2_path}: {e}")
        return None  # 返回 None 表示处理失败

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Calculate SSIM scores for images in a JSON file.")
    parser.add_argument("--file-path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output-file-path", type=str, required=True, help="Path to save the output JSON file.")
    args = parser.parse_args()
    
    # 读取 JSON 文件
    with open(args.file_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)

    for data in tqdm(datas):
        images = data['images']
        valid_images = [img for img in images if img is not None]
        
        clip_score_total = 0.0
        clip_success = True  # 标记该数据的处理是否成功
        for (img1, img2) in itertools.combinations(valid_images, 2):
            img1 = img1.replace("/mnt/workspace/zwq_data/", "/mnt/data/zwq_data/")
            img2 = img2.replace("/mnt/workspace/zwq_data/", "/mnt/data/zwq_data/")
            score = compare_images(img1, img2)
            if score is None:  # 如果处理失败
                clip_score_total = 0.0
                clip_success = False
                break
            score = (score+1)/2
            clip_score_total += score
            
        if clip_success > 0:
            data['clip_score'] = clip_score_total
            data['clip_success'] = True
        else:
            data['clip_score'] = 0.0
            data['clip_success'] = False

    # 保存处理后的数据到输出文件
    with open(args.output_file_path, 'w', encoding='utf-8') as file:
        json.dump(datas, file, ensure_ascii=False, indent=4)
