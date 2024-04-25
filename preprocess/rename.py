import os
from PIL import Image

# 指定图像文件所在的路径
image_dir = '/mnt/data/ly24/cgl/saliencymaps/train_saliencymaps_isnet'

# 获取该路径下所有文件的列表
file_list = os.listdir(image_dir)

# 遍历文件列表
for i, file_name in enumerate(file_list):
    # 构造文件的完整路径
    file_path = os.path.join(image_dir, file_name)

    # 构造新的文件名
    new_name = file_name.replace("_mask", "")
    new_path = os.path.join(image_dir, new_name)

    # 重命名文件
    os.rename(file_path, new_path)
    print(f'Renamed {file_name} to {new_name}')