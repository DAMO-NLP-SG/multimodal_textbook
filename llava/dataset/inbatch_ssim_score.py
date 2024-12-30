import json
import argparse
import cv2
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import itertools

def resize_image(image, target_size):
    # 将图片 resize 成指定大小
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def compare_images(image1_path, image2_path):
    # 
    try:
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        
        if image1 is None or image2 is None:
            print(f"Error reading images: {image1_path}, {image2_path}")
            return None  # 返回 None 表示处理失败

        # 检查并调整图像大小
        if image1.shape != image2.shape:
            print("图片尺寸不一致, 正在调整尺寸...")
            image2 = resize_image(image2, (image1.shape[1], image1.shape[0]))
        
        # 转换为灰度图像
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # 计算结构相似性指数（SSIM）
        score, _ = ssim(gray1, gray2, full=True)
        return score
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
        
        ssim_score_total = 0.0
        ssim_success = True  # 标记该数据的处理是否成功
        for (img1, img2) in itertools.combinations(valid_images, 2):
            # img1 = img1.replace("/mnt/workspace/zwq_data/", "/mnt/data/zwq_data/")
            # img2 = img2.replace("/mnt/workspace/zwq_data/", "/mnt/data/zwq_data/")
            score = compare_images(img1, img2)
            if score is None:  # 如果处理失败
                ssim_score_total = 0.0
                ssim_success = False
                break
            ssim_score_total += score
            
        if ssim_success > 0:
            data['ssim_score'] = ssim_score_total
            data['ssim_success'] = True
        else:
            data['ssim_score'] = 0.0
            data['ssim_success'] = False


    # 保存处理后的数据到输出文件
    with open(args.output_file_path, 'w', encoding='utf-8') as file:
        json.dump(datas, file, ensure_ascii=False, indent=4)
