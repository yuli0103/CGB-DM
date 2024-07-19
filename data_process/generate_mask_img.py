import cv2
import pandas as pd
import numpy as np
import os

def get_mask_image():
    input_dir = ''
    output_dir = ''

    boxes = pd.read_csv('Dataset/cgl/csv/train.csv')
    image_boxes = {}
    width = 513
    height = 750

    for _, row in boxes.iterrows():
        image_name = os.path.basename(row['poster_path'])
        box_coords = eval(row['box_elem'])

        if image_name in image_boxes:
            image_boxes[image_name].append(box_coords)
        else:
            image_boxes[image_name] = [box_coords]

    for image_name, boxes in image_boxes.items():
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        mask = np.zeros_like(image)

        for box_coords in boxes:
            x1, y1, x2, y2 = box_coords
            # Expand the range of the mask
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(width, x2 + 10)
            y2 = min(height, y2 + 10)

            mask[y1:y2, x1:x2] = 255

        mask_path = os.path.join(output_dir, image_name)
        cv2.imwrite(mask_path, mask)



