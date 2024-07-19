import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pandas import read_csv
from PIL import Image
from utils.util import box_xyxy_to_cxcywh, natural_sort_cmp

class train_dataset(Dataset):
    def __init__(self, cfg):
        img = os.listdir(cfg.paths.train.inp_dir)
        # torch.save(img, "imgname_train_save.pt")
        self.inp = list(map(lambda x: os.path.join(cfg.paths.train.inp_dir, x), img))
        self.sal = list(map(lambda x: os.path.join(cfg.paths.train.sal_dir, x), img))
        self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.train.sal_sub_dir, x), img))
        self.box_path = cfg.paths.train.salbox_dir
        self.num_class = cfg.num_class
        df_anno = read_csv(cfg.paths.train.annotated_dir)
        df_salbox = read_csv(self.box_path)
        self.max_elem = cfg.max_elem

        self.poster_name = list(img)

        self.groups_annotated = df_anno.groupby(df_anno.poster_path)
        self.groups_salbox = df_salbox.groupby(df_salbox.poster_path)
        
        self.transform_rgb = transforms.Compose([
            transforms.Resize([384,256]), # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.transform_l = transforms.Compose([
            transforms.Resize([384,256]),  # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
        ])
        
    def __len__(self):
        return len(self.inp)
     
    def __getitem__(self, idx):
        img_inp = Image.open(self.inp[idx]).convert("RGB")
        width, height = img_inp.size
        img_sal = Image.open(self.sal[idx]).convert("L")
        img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
        img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))
        img_inp = self.transform_rgb(img_inp)
        img_sal_map = self.transform_l(img_sal_map)
        imgs = torch.concat([img_inp, img_sal_map])

        sliced_df_annotated = self.groups_annotated.get_group(self.poster_name[idx])
        sliced_df_salbox = self.groups_salbox.get_group(self.poster_name[idx])

        gt_box = torch.tensor(list(map(eval, sliced_df_annotated["box_elem"])))
        gt_cls = list(sliced_df_annotated["cls_elem"])

        sal_box = torch.tensor(list(map(eval, sliced_df_salbox["box_elem"])))
        sal_box = box_xyxy_to_cxcywh(sal_box)
        sal_box[::2] /= width
        sal_box[1::2] /= height

        label_cls = np.zeros((self.max_elem, self.num_class))
        label_box = np.zeros((self.max_elem, 4))
        # label_cls_gt = np.zeros((self.max_elem, 1))
        # label_box_gt = np.zeros((self.max_elem, 4))

        for i in range(len(gt_cls)):
            if i>=self.max_elem:
                break
            label_cls[i][int(gt_cls[i])] = 1
            label_box[i] = gt_box[i]
            if label_box[i][0] > label_box[i][2] or label_box[i][1] > label_box[i][3]:
                label_box[i][:2], label_box[i][2:] = label_box[i][2:], label_box[i][:2]
            label_box[i] = box_xyxy_to_cxcywh(torch.tensor(label_box[i]))
            label_box[i][::2] /= width
            label_box[i][1::2] /= height

        for i in range(len(gt_cls), self.max_elem):
            label_cls[i][0] = 1
            # label_cls_gt[i][0] = 0

        label = np.concatenate((label_cls, label_box), axis=1)
        # label_gt = np.concatenate((label_cls_gt, label_box_gt), axis=1)

        label[:, self.num_class:] = 2 * (label[:, self.num_class:] - 0.5)
        sal_box = 2 * (sal_box - 0.5)
        return imgs, torch.tensor(label).float(), sal_box.float()


class test_uncond_dataset(Dataset):
    def __init__(self, cfg):
        img = os.listdir(cfg.paths.test.inp_dir)
        # img.sort(key=cmp_to_key(natural_sort_cmp))
        torch.save(img, cfg.imgname_order_dir)

        self.bg = list(map(lambda x: os.path.join(cfg.paths.test.inp_dir, x), img))
        self.sal = list(map(lambda x: os.path.join(cfg.paths.test.sal_dir, x), img))
        self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.test.sal_sub_dir, x), img))
        self.max_elem = cfg.max_elem

        df_salbox = read_csv(cfg.paths.test.salbox_dir)
        self.groups_salbox = df_salbox.groupby(df_salbox.poster_path)
        self.poster_name = list(img)

        self.transform_rgb = transforms.Compose([
            transforms.Resize([384, 256]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.transform_l = transforms.Compose([
            transforms.Resize([384, 256]),
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
        imgs = torch.concat([img_bg, img_sal_map])

        sliced_df_salbox = self.groups_salbox.get_group(self.poster_name[idx])
        sal_box = torch.tensor(list(map(eval, sliced_df_salbox["box_elem"])))
        sal_box = box_xyxy_to_cxcywh(sal_box)
        sal_box[::2] /= width
        sal_box[1::2] /= height
        sal_box = 2 * (sal_box - 0.5)

        return imgs, sal_box.float()

class test_cond_dataset(Dataset):
    def __init__(self, cfg):
        img = os.listdir(cfg.paths.test.inp_dir)
        torch.save(img, cfg.imgname_order_dir)

        self.inp = list(map(lambda x: os.path.join(cfg.paths.test.inp_dir, x), img))
        self.sal = list(map(lambda x: os.path.join(cfg.paths.test.sal_dir, x), img))
        self.sal_sub = list(map(lambda x: os.path.join(cfg.paths.test.sal_sub_dir, x), img))
        self.box_path = cfg.paths.test.salbox_dir
        self.num_class = cfg.num_class
        df_anno = read_csv(cfg.paths.test.annotated_dir)
        df_salbox = read_csv(self.box_path)
        self.max_elem = cfg.max_elem

        self.poster_name = list(img)

        self.groups_annotated = df_anno.groupby(df_anno.poster_path)
        self.groups_salbox = df_salbox.groupby(df_salbox.poster_path)

        self.transform_rgb = transforms.Compose([
            transforms.Resize([384, 256]),  # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.transform_l = transforms.Compose([
            transforms.Resize([384, 256]),  # (360,240)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
        ])

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        img_inp = Image.open(self.inp[idx]).convert("RGB")
        width, height = img_inp.size
        img_sal = Image.open(self.sal[idx]).convert("L")
        img_sal_sub = Image.open(self.sal_sub[idx]).convert("L")
        img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))
        img_inp = self.transform_rgb(img_inp)
        img_sal_map = self.transform_l(img_sal_map)
        imgs = torch.concat([img_inp, img_sal_map])

        sliced_df_annotated = self.groups_annotated.get_group(self.poster_name[idx])
        sliced_df_salbox = self.groups_salbox.get_group(self.poster_name[idx])

        gt_box = torch.tensor(list(map(eval, sliced_df_annotated["box_elem"])))
        gt_cls = list(sliced_df_annotated["cls_elem"])

        sal_box = torch.tensor(list(map(eval, sliced_df_salbox["box_elem"])))
        sal_box = box_xyxy_to_cxcywh(sal_box)
        sal_box[::2] /= width
        sal_box[1::2] /= height

        label_cls = np.zeros((self.max_elem, self.num_class))
        label_box = np.zeros((self.max_elem, 4))
        # label_cls_gt = np.zeros((self.max_elem, 1))
        # label_box_gt = np.zeros((self.max_elem, 4))

        for i in range(len(gt_cls)):
            if i >= self.max_elem:
                break
            label_cls[i][int(gt_cls[i])] = 1
            label_box[i] = gt_box[i]
            if label_box[i][0] > label_box[i][2] or label_box[i][1] > label_box[i][3]:
                label_box[i][:2], label_box[i][2:] = label_box[i][2:], label_box[i][:2]
            label_box[i] = box_xyxy_to_cxcywh(torch.tensor(label_box[i]))
            label_box[i][::2] /= width
            label_box[i][1::2] /= height

        for i in range(len(gt_cls), self.max_elem):
            label_cls[i][0] = 1
            # label_cls_gt[i][0] = 0

        label = np.concatenate((label_cls, label_box), axis=1)
        # label_gt = np.concatenate((label_cls_gt, label_box_gt), axis=1)

        label[:, self.num_class:] = 2 * (label[:, self.num_class:] - 0.5)
        sal_box = 2 * (sal_box - 0.5)
        return imgs, torch.tensor(label).float(), sal_box.float()

# for i in range(len(box)):
#     gtbox = box[i]
#     left, top, right, bottom = gtbox
#     left = float(left) / float(width)
#     right = float(right) / float(width)
#     top = float(top) / float(height)
#     bottom = float(bottom) / float(height)
#     left, right = clamp_w_tol(left), clamp_w_tol(right)
#     top, bottom = clamp_w_tol(top), clamp_w_tol(bottom)
#     left, right = _compare(left, right)
#     top, bottom = _compare(top, bottom)
#     center_x = clamp_w_tol((left + right) / 2)
#     center_y = clamp_w_tol((top + bottom) / 2)
#     boxwidth, boxheight = right - left, bottom - top
#     if has_valid_area(boxwidth, boxheight):
#         label_cls_gt[i][0] = int(cls[i])
#         label_box_gt[i] = [center_x, center_y, boxwidth, boxheight]

