import os
import csv
import shutil

# 读取CSV文件
data = []
with open('/home/ly24/code/py_model/Dataset/cgl/csv/train.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# 设置原始数据集目录和新目录
original_dataset_dir = '/mnt/data/ly24/train_imgs'
new_dataset_dir = '/mnt/data/ly24/train'

# 创建新目录(如果不存在)
os.makedirs(new_dataset_dir, exist_ok=True)

# 用于记录已处理过的文件名
processed_files = set()
res = 0
for item in data:
    old_filename = item['file_name']
    new_filename = item['poster_path']

    # 检查旧文件名是否已经处理过
    if old_filename in processed_files:
        # print(f"File {old_filename} has already been processed, skipping.")
        continue

    # 去掉"val/"前缀
    if new_filename.startswith('train/'):
        new_filename = new_filename[6:]

    old_path = os.path.join(original_dataset_dir, old_filename)
    new_path = os.path.join(new_dataset_dir, new_filename)


    # 检查旧文件名是否存在
    if os.path.exists(old_path):
        # 移动文件到新目录
        shutil.move(old_path, new_path)
        # 记录已处理过的文件名
        processed_files.add(old_filename)
    else:
        print(f"File {old_filename} not found, skipping.")
