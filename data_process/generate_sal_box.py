import os
import pandas as pd
import cv2
from natsort import natsorted

# Function to find the bounding box
def find_bounding_box(image_path, ifpath=True):
    if ifpath:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_path

    _, thresh = cv2.threshold(image, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, 0, 0)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, x+w, y+h)


# Function to process all images in a dataset
def process_dataset(dataset_path, output_dir):
    # List to store the results
    results = []

    # Iterate over all files in the dataset directory
    for filename in natsorted(os.listdir(dataset_path)):
        image_path = os.path.join(dataset_path, filename)
        bounding_box = find_bounding_box(image_path)
        results.append({
            "poster_path": filename,
            "box_elem": bounding_box
        })

    # Create a DataFrame and save to csv
    df = pd.DataFrame(results)
    # Output CSV file path
    df.to_csv(output_dir, index=False)

def draw_bounding_boxes(image_rgb, bounding_boxes, color=(0, 0, 255), thickness=2):
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = bounding_boxes
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, thickness)

def main():
    '''generate saliency bounding box csv file'''
    # Replace with your saliency image dataset directory path
    input_dir = ''
    output_dir = ''
    process_dataset(input_dir, output_dir)

    '''visualize saliency bounding box'''
    # image = cv2.imread("", cv2.IMREAD_GRAYSCALE)
    # if image is None:
    #     raise ValueError("Image not found, please check the image path.")
    # bounding_box = find_bounding_box(image)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # draw_bounding_boxes(image_rgb, bounding_boxes, color=(0, 0, 255), thickness=3)
    # # for x1, y1, x2, y2 in bounding_boxes:
    # #     cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 0, 255), 5)
    # cv2.imwrite("image_with_bounding_box.png", image_rgb)


if __name__ == "__main__":
    main()
