import cv2
import math
# 显著性检测计算主题对象的占比
def calc_sal_ratio(img):
    thresh = 25
    ret, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    foreground_pixels = cv2.countNonZero(thresh_img)
    total_pixels = thresh_img.shape[0] * thresh_img.shape[1]
    ratio = foreground_pixels / total_pixels
    return ratio

def calc_image_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()
    total_pixels = sum(hist)
    entropy = 0
    for px_val in hist:
        px_prob = px_val / total_pixels
        if px_prob > 0:
            entropy += px_prob * math.log2(px_prob)

    return -entropy


if __name__ == '__main__':
    img_path = "/home/ly24/code/py_model/Dataset/pku/test/image_canvas/133.png"
    img = cv2.imread(img_path, 0)
    ratio = calc_image_entropy(img)
    print(ratio)