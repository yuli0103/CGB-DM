import pandas as pd

df = pd.read_csv('/mnt/data/ly24/dataset/cgl/split/csv/test_unanno_sal.csv')
df['poster_path'] = df['poster_path'].str.replace('test/', '', regex=False)
# df['poster_path'] = df['poster_path'].str.replace('jpg', 'png', regex=False)
df.to_csv('/mnt/data/ly24/dataset/cgl/split/csv/test_unanno_sal.csv', index=False)
print("处理完成")

import os


# def rename_jpg_to_png(folder_path):
#     # 确保文件夹路径存在
#     if not os.path.exists(folder_path):
#         print(f"文件夹 {folder_path} 不存在")
#         return
#
#     # 遍历文件夹中的所有文件
#     for filename in os.listdir(folder_path):
#         # 检查文件是否以 .jpg 结尾（不区分大小写）
#         if filename.lower().endswith('.jpg'):
#             # 构建旧文件的完整路径
#             old_file = os.path.join(folder_path, filename)
#
#             # 构建新文件名（将 .jpg 替换为 .png）
#             new_filename = os.path.splitext(filename)[0] + '.png'
#             new_file = os.path.join(folder_path, new_filename)
#
#             # 重命名文件
#             os.rename(old_file, new_file)
#             print(f"已将 {filename} 重命名为 {new_filename}")
#
#
# # 使用示例
# folder_path = '/mnt/data/ly24/dataset/cgl/split/train/inpaint'  # 替换为你的文件夹路径
# rename_jpg_to_png(folder_path)