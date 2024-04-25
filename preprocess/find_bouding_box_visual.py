import cv2
from matplotlib import pyplot as plt

def find_bounding_box(image):
    if image is None:
        raise ValueError("Image not loaded properly.")

    _, thresh = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, 0, 0)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, x+w, y+h)

def main():
    # 读取图片，使用cv2.IMREAD_COLOR确保载入图片包含颜色信息
    image = cv2.imread("/mnt/data/kl23/pku/nosplit/train/saliency/86.png", cv2.IMREAD_GRAYSCALE)

    # 确保图片加载正确
    if image is None:
        raise ValueError("没有找到图像，请检查图像路径。")

    x1, y1, x2, y2 = find_bounding_box(image)
    print(x1, y1, x2, y2)
    # 转换颜色空间从BGR到RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 在图像上绘制边界框，红色，2px宽度
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite("image_with_bounding_box.png", image_rgb)
    print("图像已保存为 image_with_bounding_box.png")
    # 使用matplotlib显示图像
    # plt.imshow(image_rgb)
    # plt.title('带有边界框的图像')
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

if __name__ == "__main__":
    main()