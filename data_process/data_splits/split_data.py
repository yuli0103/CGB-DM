import os
import shutil
import pandas as pd

def split_data():
    with open('data_splits/pku/train.txt', 'r') as f:
        numbers = [line.strip() for line in f]

    source_dir = ''
    target_dir = ''
    os.makedirs(target_dir, exist_ok=True)
    cnt = 0

    for filename in os.listdir(source_dir):

        prefix = filename.split('.')[0]
        if prefix in numbers:
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            shutil.copy(src_path, dst_path)
            print(f"Moved {filename} to {target_dir}")
            cnt += 1

    print(cnt)


