import os
import pandas as pd
import cv2
from natsort import natsorted  # 引入natsort库,用于按自然顺序排序

# Function to find the bounding box
def find_bounding_box(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not loaded properly.")

    _, thresh = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, 0, 0)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, x+w, y+h)

# Function to process all images in a dataset
def process_dataset(dataset_path):
    # List to store the results
    results = []

    # Iterate over all files in the dataset directory
    for filename in natsorted(os.listdir(dataset_path)):  # 使用natsorted对文件名进行自然排序
        if filename.endswith('.png'):  # Assuming the images are in PNG format
            image_path = os.path.join(dataset_path, filename)
            bounding_box = find_bounding_box(image_path)
            results.append({
                "poster_path": "test/" + filename,
                "box_elem": bounding_box
            })

    # Create a DataFrame and save to csv
    df = pd.DataFrame(results)
    csv_path = '/mnt/data/ly24/Dataset/cgl/csv/cgltest_saliency_isnet_box_thresh25.csv'  # Output CSV file path
    df.to_csv(csv_path, index=False)

    return csv_path

# Specify your dataset directory path
dataset_directory_path = '/mnt/data/ly24/Dataset/cgl/saliencymaps/test_saliencymaps_isnet'  # Replace with your dataset directory path
output_csv_path = process_dataset(dataset_directory_path)