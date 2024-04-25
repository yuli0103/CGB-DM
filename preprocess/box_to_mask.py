import cv2
import pandas as pd
import numpy as np
import os

# 读取CSV文件
boxes = pd.read_csv('Dataset/cgl/csv/train.csv')
image_boxes = {}

# 遍历CSV文件,收集每个图像的所有边界框坐标
for _, row in boxes.iterrows():
    image_name = os.path.basename(row['poster_path'])
    box_coords = eval(row['box_elem'])

    if image_name in image_boxes:
        image_boxes[image_name].append(box_coords)
    else:
        image_boxes[image_name] = [box_coords]

# 遍历字典,为每个图像生成对应的掩码图像
for image_name, boxes in image_boxes.items():
    # 读取原始图像
    image_path = os.path.join('/mnt/data/ly24/train', image_name)
    image = cv2.imread(image_path)

    # 创建掩码图像
    mask = np.zeros_like(image)

    # 在掩码图上绘制所有边界框
    for box_coords in boxes:
        x1, y1, x2, y2 = box_coords

        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(513, x2 + 10)
        y2 = min(750, y2 + 10)

        mask[y1:y2, x1:x2] = 255

    # 保存掩码图像
    # mask_name = os.path.splitext(image_name)[0] + '_mask.png'
    mask_path = os.path.join('/mnt/data/ly24/mask/train', image_name)
    cv2.imwrite(mask_path, mask)

