import os
import csv
import cv2
import torch
import numpy as np
from collections import defaultdict

# 导入 Detectron2 工具包
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 加载 Detectron2 预训练模型配置
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 # 设置阈值以保留置信度高的检测结果
cfg.MODEL.WEIGHTS = "/home/ly24/code/py_model/model_weight/model_final_68b088.pkl"
device_id = 0
cfg.MODEL.DEVICE = f"cuda:{device_id}"

# 创建预测器
predictor = DefaultPredictor(cfg)

# 数据集路径
dataset_path = "/home/ly24/code/py_model/Dataset/train/inpainted_poster"

# 创建用于存储结果的字典
results = defaultdict(list)

# 遍历数据集中的图像
for filename in os.listdir(dataset_path):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join(dataset_path, filename)

        # 加载图像
        img = cv2.imread(image_path)

        # 运行模型进行预测
        outputs = predictor(img)

        # 从输出中提取检测框
        instances = outputs["instances"]
        boxes = instances.pred_boxes.tensor.cpu().numpy()

        # 将检测框结果存储在字典中
        filename_without_mask = filename.replace("_mask.png", "")
        if len(boxes) == 0:
            results[filename_without_mask].append([0, [0, 0, 0, 0]])
        else:
            for box in boxes:
                cls_elem = 0
                box_elem = [int(x) for x in box]
                results[filename_without_mask].append([cls_elem, box_elem])

# 将结果按照文件名中的数字进行排序
sorted_results = sorted(results.items(), key=lambda x: int(x[0]))

# 将结果保存到CSV文件
with open("detection_results_train_4.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"poster_path", "total_elem", "cls_elem", "box_elem"])  # 写入标题行
    for filename, boxes in sorted_results:
        poster_path = f"train/{filename}"
        total_elem = len(boxes)
        for cls_elem, box_elem in boxes:
            writer.writerow([f"{poster_path}.png", total_elem, cls_elem, box_elem])