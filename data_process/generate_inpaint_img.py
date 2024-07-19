import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import torch

def lama_inpaint():
    input_dir = ''
    mask_dir = ''
    output_dir = ''

    device_id = 2
    device_str = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inpainting = pipeline(Tasks.image_inpainting, model='damo/cv_fft_inpainting_lama', refine=True, device=device_str)

    for filename in os.listdir(input_dir):
        input_location = os.path.join(input_dir, filename)
        mask_location = os.path.join(mask_dir, filename)
        input = {
            'img': input_location,
            'mask': mask_location,
        }

        result = inpainting(input)
        vis_img = result[OutputKeys.OUTPUT_IMG]

        _, ext = os.path.splitext(filename)

        if ext.lower() == '.jpg':
            new_filename = filename.replace(ext, '_mask.jpg')
        elif ext.lower() == '.png':
            new_filename = filename.replace(ext, '_mask.png')
        else:
            continue

        output_location = os.path.join(output_dir, new_filename)
        cv2.imwrite(output_location, vis_img)

    print("Inpainting completed!")