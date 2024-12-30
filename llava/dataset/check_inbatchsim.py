import json


# file1 = '/mnt/data/zwq_data/interleaved_dataset/mmc4_core_ff_as_obelics_format/mmc4_filter_length_as_obelics_format_67w_length_7k_image_larger_3_sub_clip1.json'

# file2 = '/mnt/data/zwq_data/interleaved_dataset/mmc4_core_ff_as_obelics_format/mmc4_filter_length_as_obelics_format_67w_length_7k_image_larger_3_sub_ssim.json'

# file1 = '/mnt/data/zwq_data/dataset_benchmark/obelics/obelics61w_image_number_3_11_sample_verify_sub_clip1.json'

# file2 = '/mnt/data/zwq_data/dataset_benchmark/obelics/obelics61w_image_number_3_11_sample_verify_sub_ssim.json'


file1 = '/mnt/data/zwq_data/interleaved_dataset/ours_as_obelics_format/ours_61w_refined_asr_ocr_image_filter_as_obelics_format_sub_clip1.json'

file2 = '/mnt/data/zwq_data/interleaved_dataset/ours_as_obelics_format/ours_61w_refined_asr_ocr_image_filter_as_obelics_format_sub_ssim.json'


with open(file1, 'r', encoding='utf-8') as file:
    datas1 = json.load(file)


with open(file2, 'r', encoding='utf-8') as file:
    datas2 = json.load(file)


for target_images_num in range(4, 12):
    score_sum = 0
    count = 0
    combination = target_images_num * (target_images_num - 1) / 2
    for i in range(len(datas1)):
        images_num = datas1[i]['image_num']
        ssim_score = datas2[i]['ssim_score']
        clip_score = datas1[i]['clip_score']
        clip_success = datas1[i]['clip_success']
        # ssim_success = datas2[i]['ssim_success']
        clip_score = (clip_score+1)/2
        # 仅当 images_num 为当前循环的目标值时计算分数
        if images_num == target_images_num and clip_success: #  and ssim_success:
            score = (ssim_score + clip_score) / 2
            score /= combination
            score_sum += score
            count += 1
    
    # 计算并打印当前 images_num 的平均分数
    if count > 0:
        average_score = score_sum / count
        print(count)
        print(f"images_num={target_images_num} 的平均分数是: {average_score}")
    else:
        print(f"images_num={target_images_num} 没有找到符合条件的数据。")


