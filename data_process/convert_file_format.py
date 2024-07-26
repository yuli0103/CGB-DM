import csv
import json
import pyarrow.parquet as pq
import torch
from pathlib import Path
import pickle
import os
import shutil

def json_to_csv():
    '''
    json to csv(cgl)
    '''
    with open('/mnt/data/ly24/layout_train_6w_fixed_v2.json', 'r') as f:
        data = json.load(f)
    image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    with open('/mnt/data/ly24/cgl_train.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['poster_path', 'total_elem', 'cls_elem', 'box_elem', 'file_name'])

        for annotation in data['annotations']:
            for bbox in annotation:
                image_id = bbox['image_id']
                category_id = bbox['category_id']
                xmin, ymin, width, height = bbox['bbox']
                box_elem = [xmin, ymin, xmin+width, ymin+height]
                file_name = image_id_to_filename[image_id]
                writer.writerow([str(image_id) + ".png", len(annotation), category_id, box_elem, file_name])

def read_parquet():
    # Reading Parquet Files
    table = pq.read_table('test-00000-of-00001.parquet')
    print(table.schema)
    for batch in table.to_batches():
        for row in batch:
            print(row)

def read_pkl():
    # Reading .pkl Files
    pklpath = ""
    with open(pklpath,'rb') as f:
        data = pickle.load(f)
        for i in range(len(data['results'])):
            print(type(data['results'][i]['label']))

def pt_to_pkl():
    '''
    .pt to .pkl
    '''

    ptpath = ""
    datas = torch.load(ptpath)
    xyz = {}
    xyz['results'] = []

    '''cgl name dictionary'''
    dic = {}
    with open('/mnt/data/ly24/Dataset/cgl/split/csv/test.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            file_name = row[4]
            poster_path = row[0]
            file_name = file_name.split('.')[0]
            poster_number = Path(poster_path).stem
            dic[poster_number] = file_name
    names = torch.load("/home/ly24/code/ditl/ptfile/cgl_split_test.pt")
    for idx, name in enumerate(names):
        number = name.split('.')[0]
        # print(number)
        new_name = dic[number]
        data = datas[idx]
        cls = data[:, 0]
        mask = (cls > 0).reshape(-1)
        layout = data[mask]

        temp_dic = {}
        temp_dic['label'] = [int(x) for x in layout[:, 0]]
        temp_dic['width'] = [float(x) for x in layout[:, 3]]
        temp_dic['height'] = [float(x) for x in layout[:, 4]]
        temp_dic['center_x'] = [float(x) for x in layout[:, 1]]
        temp_dic['center_y'] = [float(x) for x in layout[:, 2]]
        temp_dic['id'] = new_name

        xyz['results'].append(temp_dic)
    print(len(xyz['results']))

def rename():
    image_dir = ''
    file_list = os.listdir(image_dir)
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(image_dir, file_name)
        new_name = file_name.replace("_mask", "")
        new_path = os.path.join(image_dir, new_name)
        os.rename(file_path, new_path)
        print(f'Renamed {file_name} to {new_name}')

def file_process():
    with open('', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    original_dataset_dir = ''
    new_dataset_dir = ''
    os.makedirs(new_dataset_dir, exist_ok=True)

    processed_files = set()
    for item in data:
        old_filename = item['file_name']
        new_filename = item['poster_path']

        if old_filename in processed_files:
            # print(f"File {old_filename} has already been processed, skipping.")
            continue

        if new_filename.startswith('train/'):
            new_filename = new_filename[6:]

        old_path = os.path.join(original_dataset_dir, old_filename)
        new_path = os.path.join(new_dataset_dir, new_filename)

        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            processed_files.add(old_filename)
        else:
            print(f"File {old_filename} not found, skipping.")

if __name__ == "__main__":
    json_to_csv()