#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 18:13:15 2022

@author: kinsleyhsu
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pandas import read_csv
from PIL import Image
from utils.designSeq import reorder_pku,reorder_cgl
from torchvision.transforms import functional as F
import pickle
import sys
import cv2
from functools import cmp_to_key
import re
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0) , (y1 - y0)]
    return torch.stack(b, dim=-1)
def natural_sort_cmp(a, b):
    """
    用于比较两个字符串的自然排序顺序
    """
    # 提取文件名中的数字部分, 匹配 .png 和 .jpg 扩展名
    a_match = re.match(r'(\d+)\.(png|jpg)$', a, re.IGNORECASE)
    b_match = re.match(r'(\d+)\.(png|jpg)$', b, re.IGNORECASE)

    if a_match and b_match:
        # 如果两个文件名都符合格式, 则比较数字部分
        a_num = int(a_match.group(1))
        b_num = int(b_match.group(1))
        return a_num - b_num
    elif a_match:
        return -1
    elif b_match:
        return 1
    else:
        return 0

class canvas_train(Dataset):
    def __init__(self, inp_dir, sal_dir, sal_sub_dir, csv_path, box_path, max_elem=16, num_class=4, dataset="pku", train_if=True):
        img = os.listdir(inp_dir)
        img.sort(key=cmp_to_key(natural_sort_cmp))
        # torch.save(img, "train_order.pt")
        self.inp = list(map(lambda x: os.path.join(inp_dir, x), img))
        if dataset== "pku":
            self.sal = list(map(lambda x: os.path.join(sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(sal_sub_dir, x), img))
        elif dataset== "cgl":
            self.sal = list(map(lambda x: os.path.join(sal_dir, x.replace(".jpg", ".png")), img))
            self.sal_sub = list(map(lambda x: os.path.join(sal_sub_dir, x.replace(".jpg", ".png")), img))
        self.ds = dataset

        self.box_path = box_path
        self.num_class = num_class
        df = read_csv(csv_path)
        df_box = read_csv(box_path)
        self.max_elem = max_elem
        if train_if:
            self.poster_name = list(map(lambda x: "train/" + x, img))
        else:
            self.poster_name = list(map(lambda x: "test/" + x, img))
        self.groups = df.groupby(df.poster_path)
        self.groups_box = df_box.groupby(df_box.poster_path)
        
        self.transform_rgb = transforms.Compose([
            transforms.Resize([256,256]), # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.transform_l = transforms.Compose([
            transforms.Resize([256,256]),  # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
        ])
        
    def __len__(self):
        return len(self.inp)
     
    def __getitem__(self, idx):
        # from PIL import ImageFile
        # ImageFile.LOAD_TRUNCATED_IMAGES = True

        img_inp = Image.open(self.inp[idx]).convert("RGB")
        width, height = img_inp.size
        img_sal = Image.open(self.sal[idx]).convert("L")
        img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
        img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))
        img_inp = self.transform_rgb(img_inp)
        img_sal_map = self.transform_l(img_sal_map)
        cc = torch.concat([img_inp, img_sal_map])
         
        label_cls = np.zeros((self.max_elem, self.num_class))
        label_box = np.zeros((self.max_elem, 4))
        sliced_df = self.groups.get_group(self.poster_name[idx])
        sliced_df_box = self.groups_box.get_group(self.poster_name[idx].replace(".jpg", ".png"))

        box = torch.tensor(list(map(eval, sliced_df["box_elem"])))
        cls = list(sliced_df["cls_elem"])
        # if self.ds == 'pku':
        #     order = reorder_pku(cls, box, self.max_elem)
        # else:
        #     order = reorder_cgl(cls, box, self.max_elem)

        box_pre = torch.tensor(list(map(eval, sliced_df_box["box_elem"])))
        box_pre = box_xyxy_to_cxcywh(box_pre)

        box_pre[::2] /= width
        box_pre[1::2] /= height

        # for i in range(len(order)):
        #     idx = order[i]
        #     label_cls[i][int(cls[idx])] = 1
        #     label_box[i] = box[idx]
        #     if label_box[i][0] > label_box[i][2] or label_box[i][1] > label_box[i][3]:
        #         label_box[i][:2], label_box[i][2:] = label_box[i][2:], label_box[i][:2]
        #     label_box[i] = box_xyxy_to_cxcywh(torch.tensor(label_box[i]))
        #     label_box[i][::2] /= width
        #     label_box[i][1::2] /= height
        # for i in range(len(order), self.max_elem):
        #     label_cls[i][0] = 1

        for i in range(len(cls)):
            if i>=self.max_elem:
                break
            label_cls[i][int(cls[i])] = 1
            label_box[i] = box[i]
            if label_box[i][0] > label_box[i][2] or label_box[i][1] > label_box[i][3]:
                label_box[i][:2], label_box[i][2:] = label_box[i][2:], label_box[i][:2]
            label_box[i] = box_xyxy_to_cxcywh(torch.tensor(label_box[i]))
            label_box[i][::2] /= width
            label_box[i][1::2] /= height
        for i in range(len(cls), self.max_elem):
            label_cls[i][0] = 1

        label = np.concatenate((label_cls, label_box), axis=1)
        
        return cc, torch.tensor(label).float(), box_pre.float()

class canvas_test(Dataset):
    def __init__(self, bg_dir, sal_dir, sal_sub_dir, test_box_path, max_elem, dataset="pku"):
        img = os.listdir(bg_dir)
        img.sort(key=cmp_to_key(natural_sort_cmp))
        # torch.save(img, "ptfile/test_order_pku_split_test.pt")
        self.bg = list(map(lambda x: os.path.join(bg_dir, x), img))
        if dataset== "pku":
            self.sal = list(map(lambda x: os.path.join(sal_dir, x), img))
            self.sal_sub = list(map(lambda x: os.path.join(sal_sub_dir, x), img))
        elif dataset== "cgl":
            self.sal = list(map(lambda x: os.path.join(sal_dir, x.replace(".jpg", ".png")), img))
            self.sal_sub = list(map(lambda x: os.path.join(sal_sub_dir, x.replace(".jpg", ".png")), img))

        self.box_path = test_box_path
        self.max_elem = max_elem

        df_box = read_csv(test_box_path)
        self.groups_box = df_box.groupby(df_box.poster_path)
        self.poster_name = list(map(lambda x: "test/" + x, img))

        self.transform_rgb = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.transform_l = transforms.Compose([
            transforms.Resize([256,256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
        ])

    def __len__(self):
        return len(self.bg)
    
    def __getitem__(self, idx):

        img_bg = Image.open(self.bg[idx]).convert("RGB")

        width, height = img_bg.size
        img_sal = Image.open(self.sal[idx]).convert("L")
        img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
        img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))
        img_bg = self.transform_rgb(img_bg)
        img_sal_map = self.transform_l(img_sal_map)
        cc = torch.concat([img_bg, img_sal_map])

        sliced_df_box = self.groups_box.get_group(self.poster_name[idx].replace(".jpg", ".png"))
        detect_box = torch.tensor(list(map(eval, sliced_df_box["box_elem"])))
        detect_box = box_xyxy_to_cxcywh(detect_box)
        detect_box[::2] /= width
        detect_box[1::2] /= height

        return cc, detect_box.float()

