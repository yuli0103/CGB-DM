import csv
import json

# 读取JSON文件
with open('Dataset/cgl/json/yinhe.json', 'r') as f:
    data = json.load(f)

# 创建CSV文件
with open('Dataset/cgl/csv/test.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # 写入标题行
    writer.writerow(['poster_path', 'file_name'])

    # 遍历images
    for image in data['images']:
        image_id = image['id']
        file_name = image['file_name']

        # 写入数据行
        writer.writerow(["test/" + str(image_id) + ".jpg", file_name])