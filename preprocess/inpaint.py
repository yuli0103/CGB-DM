import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import torch

# 定义输入和输出路径
input_dir = '/mnt/data/ly24/train'
mask_dir = '/mnt/data/ly24/mask/train'
output_dir = '/mnt/data/ly24/inpaint/train'


# 设置要使用的GPU设备ID
device_id = 2  # 根据你的实际情况设置
device_str = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

# 创建输出目录(如果不存在)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama', refine=True,device=device_str)

# 遍历输入目录中的所有图像
for filename in os.listdir(input_dir):
    # 构建输入路径
    input_location = os.path.join(input_dir, filename)
    mask_location = os.path.join(mask_dir, filename)

    # 构建输入字典
    input = {
        'img': input_location,
        'mask': mask_location,
    }

    # 执行inpainting
    result = inpainting(input)
    vis_img = result[OutputKeys.OUTPUT_IMG]


    # 获取文件扩展名
    _, ext = os.path.splitext(filename)

    # 根据扩展名构建目标文件名
    if ext.lower() == '.jpg':
        new_filename = filename.replace(ext, '_mask.jpg')
    elif ext.lower() == '.png':
        new_filename = filename.replace(ext, '_mask.png')
    else:
        continue  # 跳过其他扩展名的文件
    output_location = os.path.join(output_dir, new_filename)

    # 保存输出图像
    cv2.imwrite(output_location, vis_img)
    # print(f"Processed {filename}")

print("Inpainting completed!")