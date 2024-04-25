import cv2
import matplotlib.pyplot as plt
import torch

# 导入 Detectron2 工具包
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 加载 Detectron2 预训练模型配置
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # 设置阈值以保留置信度高的检测结果
cfg.MODEL.WEIGHTS = "/home/ly24/code/py_model/model_weight/model_final_68b088.pkl"

# 创建预测器
predictor = DefaultPredictor(cfg)

# 加载图像
img = cv2.imread("/home/ly24/code/py_model/Dataset/pku/test/saliencymaps_pfpn/0.png")

# 运行模型进行预测
outputs = predictor(img)

# 在图像上绘制检测结果
v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out_img = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]

# 保存检测后的图像
cv2.imwrite("../detected_image.jpg", out_img)

# 显示结果
# plt.figure(figsize=(10, 8))
# plt.imshow(out_img)
# plt.axis('off')
# plt.show()