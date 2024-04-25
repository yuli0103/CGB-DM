import csv
import json

# 读取JSON文件
with open('Dataset/cgl/json/layout_train_6w_fixed_v2.json', 'r') as f:
    data = json.load(f)

# 创建一个字典用于存储每个image_id对应的文件名
image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}

# 创建CSV文件
with open('Dataset/cgl/csv/train.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    # 写入标题行
    writer.writerow(['poster_path', 'total_elem', 'cls_elem', 'box_elem', 'file_name'])

    # 遍历annotations
    for annotation in data['annotations']:
        for bbox in annotation:
            image_id = bbox['image_id']
            category_id = bbox['category_id']
            xmin, ymin, width, height = bbox['bbox']
            box_elem = [xmin, ymin, xmin+width, ymin+height]

            # 获取文件名
            file_name = image_id_to_filename[image_id]

            # 写入数据行
            writer.writerow(["train/" + str(image_id) + ".jpg", len(annotation), category_id, box_elem, file_name])