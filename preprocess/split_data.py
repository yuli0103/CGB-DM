import os
import shutil

# 读取文本文件中的数字
with open('splits/pku/val.txt', 'r') as f:
    numbers = [line.strip() for line in f]

# 当前目录
source_dir_1 = '/mnt/data/kl23/pku/nosplit/train/inpainted_images'
# source_dir_2 = '/mnt/data/ly24/Dataset/cgl/saliencymaps/val_saliencymaps_isnet'

# 目标目录
target_dir = '/mnt/data/kl23/pku/split/val/inpaint'
os.makedirs(target_dir, exist_ok=True)
cnt=0
# 遍历当前目录下的所有文件
for filename in os.listdir(source_dir_1):

    # 获取文件名前缀
    prefix = filename.split('.')[0]
    # 检查前缀是否在文本文件中
    if prefix in numbers:
        # 构建源文件路径和目标文件路径
        src_path = os.path.join(source_dir_1, filename)
        dst_path = os.path.join(target_dir, filename)
        # 移动文件
        shutil.copy(src_path, dst_path)
        print(f"Moved {filename} to {target_dir}")
        cnt += 1

# for filename in os.listdir(source_dir_2):
#
#     # 获取文件名前缀
#     prefix = filename.split('.')[0]
#     # 检查前缀是否在文本文件中
#     if prefix in numbers:
#         # 构建源文件路径和目标文件路径
#         src_path = os.path.join(source_dir_2, filename)
#         dst_path = os.path.join(target_dir, filename)
#         # 移动文件
#         shutil.copy(src_path, dst_path)
#         print(f"Moved {filename} to {target_dir}")
#         cnt += 1
print(cnt)

# import pandas as pd
#
# # 读取文本文件中的数字
# with open('splits/cgl_x/train.txt', 'r') as f:
#     numbers = [line.strip() for line in f]
#
# # 读取原始CSV文件
# original_df = pd.read_csv('/mnt/data/ly24/Dataset/cgl/csv/sal_all.csv')
#
# # 创建一个空的DataFrame作为目标
# rows_to_keep = []
# cnt = 0
# # 遍历原始DataFrame的每一行
# for index, row in original_df.iterrows():
#     # 获取poster_path值
#     poster_path = row['poster_path']
#     new_poster_path = 'train/' + poster_path.split('/')[1]
#     # 检查poster_path是否符合要求
#     if poster_path.split('/')[1].split('.')[0] in numbers:
#         # 如果符合要求,则将该行添加到目标DataFrame
#         row['poster_path'] = new_poster_path
#         rows_to_keep.append(row)
#         cnt += 1
# # 将列表转换为新的DataFrame
# target_df = pd.DataFrame(rows_to_keep, columns=original_df.columns)
#
# # 将目标DataFrame写入新的CSV文件
# target_df.to_csv('/mnt/data/ly24/Dataset/cgl/split/csv/train_sal.csv', index=False)
# print(cnt)
