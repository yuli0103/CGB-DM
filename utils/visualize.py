import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import os
from utils.util import box_cxcywh_to_xyxy
import numpy as np
import seaborn as sns
import csv
from pathlib import Path
import pickle

def get_colors(n_colors: int) -> list[tuple[int, int, int]]:
    colors = sns.color_palette("husl", n_colors=n_colors)
    colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
    return colors  # type: ignore

def draw_box(img, elems, num_class):
    drawn_fill = img.copy()
    draw_f = ImageDraw.Draw(drawn_fill, "RGBA")

    if num_class==4:
        colors = get_colors(3)
        colors[0], colors[1] = colors[1], colors[0]
    else:
        colors = get_colors(4)
        emb, logo, text, und = colors
        colors = (text, logo, und, emb)

    s_elems = sorted(list(elems), key=lambda x: x[0], reverse=True)
    for cls, box in s_elems:
        if cls:
            # c_fill = colors[int(cls)]
            label = int(cls) - 1
            c_fill = colors[label] + (160,)
            draw_f.rectangle(tuple(box), outline=colors[label], fill=c_fill)

    return drawn_fill

def draw_image(box, cls, img, img_name, width, height, numclass, save_dir):
    cls = cls.detach().cpu().numpy()
    if cls.ndim>=3:
        cls = cls.squeeze(0)
    if box.ndim>=3:
        box = box.squeeze(0)
    box = torch.clamp(box_cxcywh_to_xyxy(box.detach().cpu()), 0, 1)
    box[:, ::2] *= width
    box[:, 1::2] *= height
    box = box.round().int().numpy()

    drawn = draw_box(img, zip(cls, box), numclass)
    save_dir_path = os.path.join(save_dir, '{}.png'.format(img_name))
    drawn.save(save_dir_path)

def visualize_images(img_names, test_output, cfg):
    clses, boxes = test_output[:, :, :1], test_output[:, :, 1:]
    save_dir = cfg.save_imgs_dir
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(test_output.shape[0]):
        image_path = os.path.join(cfg.paths.test.inp_dir, img_names[idx])
        img = Image.open(image_path).convert("RGB")
        box, cls = boxes[idx], clses[idx]
        draw_image(box, cls, img, img_names[idx], cfg.width, cfg.height, cfg.num_class, save_dir)
        print(idx)

def draw_bgwhite_image(img_names, test_output, cfg):
    clses, boxes = test_output[:, :, :1], test_output[:, :, 1:]
    save_dir = cfg.save_imgs_dir
    os.makedirs(save_dir, exist_ok=True)
    width, height = cfg.width, cfg.height

    for idx in range(test_output.shape[0]):
        img_white = Image.new('RGB', (width, height), color='white')
        box, cls = boxes[idx], clses[idx]
        draw_image(box, cls, img_white, img_names[idx], width, height, cfg.num_class, save_dir)
        print(idx)

def draw_frompkl_image(pklpath, width, height, num_class, save_dir, test_bg_dir):
    dic = {}
    with open('', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            file_name = row[4]
            poster_path = row[0]

            file_name = file_name.split('.')[0]
            poster_number = Path(poster_path).stem.split('.')[-1]
            dic[file_name] = poster_number

    os.makedirs(save_dir, exist_ok=True)
    with open(pklpath,'rb') as f:
        data = pickle.load(f)
        cnt=0
        for result in data['results']:
            label, w, h, xc, yc = result['label'], result['width'], result['height'], result['center_x'], result['center_y']
            imgid = result['id']
            imgid = dic[imgid]
            boxes = []
            clses = []
            if num_class == 4:
                for i in range(len(label)):
                    if label[i] == 0:
                        clses.append(2)
                    elif label[i] == 2:
                        clses.append(3)
                    else:
                        clses.append(1)
            else:
                for i in range(len(label)):
                    if label[i] == 2:
                        clses.append(1)
                    elif label[i] == 3:
                        clses.append(3)
                    elif label[i] == 1:
                        clses.append(2)
                    else:
                        clses.append(4)

            for i in range(len(clses)):
                boxes.append([xc[i], yc[i], w[i], h[i],])
            clses = torch.tensor(clses)
            boxes = torch.tensor(boxes)
            if len(clses) == 0:
                continue
            image_path = os.path.join(test_bg_dir, imgid+'.jpg')
            img = Image.open(image_path).convert("RGB")
            draw_image(boxes, clses, img, imgid, width, height, num_class, save_dir)
            cnt += 1
            print(cnt)