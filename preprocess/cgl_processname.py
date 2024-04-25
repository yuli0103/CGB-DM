import pandas as pd

# 读取CSV文件
df = pd.read_csv('/mnt/data/ly24/Dataset/cgl/csv/train_all.csv')

# 创建一个字典,存储每个poster_path对应的x值
poster_path_dict = {}
for index, row in df.iterrows():
    poster_path = row['poster_path']
    file_name = row['file_name'].split('.')[0]

    x = poster_path.split('/')[1].split('.')[0]
    poster_path_dict[file_name] = x
print(f"Number of key-value pairs in poster_path_dict: {len(poster_path_dict)}")
# 读取文本文件中的字符串
with open('splits/cgl/val.txt', 'r') as f:
    input_strings = [line.strip() for line in f]
print(f"Number of strings in input_strings: {len(input_strings)}")
# 存储转换后的名字
output_strings = []

cnt=0
# 遍历输入字符串,查找对应的x值
for input_string in input_strings:
    if input_string in poster_path_dict:
        x = poster_path_dict[input_string]
        output_strings.append(x)
        cnt += 1
print(cnt)

# 将转换后的名字写入新的文本文件
with open('splits/cgl_x/val.txt', 'w') as f:
    for output_string in output_strings:
        f.write(output_string + '\n')