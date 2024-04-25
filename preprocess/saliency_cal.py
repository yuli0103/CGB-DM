from PIL import Image
import numpy as np
import cv2

img_path = "/home/ly24/code/py_model/Dataset/pku/test/saliencymaps_pfpn/212.png"
img = cv2.imread(img_path, 0)
# 设置阈值,将像素分为前景和背景
thresh = 25
ret, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

# 统计前景像素数
foreground_pixels = cv2.countNonZero(thresh_img)

# 获取总像素数
total_pixels = thresh_img.shape[0] * thresh_img.shape[1]

# 计算比例
ratio = foreground_pixels / total_pixels

print(f"主要物体占据的空间比例: {ratio * 100:.2f}%")