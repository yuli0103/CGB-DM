#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 03:23:42 2022

@author: kinsleyhsu
"""
import sys

import torch
import numpy as np
import random
import torch.nn as nn
import os
import fsspec
from torchvision.ops.boxes import box_area
from collections import OrderedDict

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0) , (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * abs(w)), (y_c - 0.5 * abs(h)),
         (x_c + 0.5 * abs(w)), (y_c + 0.5 * abs(h))]
    return torch.stack(b, dim=-1)
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def save_model(model: nn.Module, ckpt_dir: str, best_or_final: str = "best"):
    model_path = os.path.join(ckpt_dir, f"{best_or_final}_model.pt")
    with fsspec.open(str(model_path), "wb") as file_obj:
        torch.save(model.state_dict(), file_obj)
    return model

# def update_ema(target_params, source_params, rate=0.99):
#     """
#     Update target parameters to be closer to those of source parameters using
#     an exponential moving average.
#
#     :param target_params: the target parameter sequence.
#     :param source_params: the source parameter sequence.
#     :param rate: the EMA rate (closer to 1 means slower).
#     """
#     for targ, src in zip(target_params, source_params):
#         targ.detach().mul_(rate).add_(src, alpha=1 - rate)

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
def box_intersection(boxes1, boxes2):
    """
    计算两组边界框之间的交集区域。
    参数:
        boxes1 (Tensor): 形状为(N, 4)的边界框坐标张量。
        boxes2 (Tensor): 形状为(M, 4)的边界框坐标张量。
    返回:
        intersections (Tensor): 形状为(N, M)的交集面积张量。
    """
    x1_max = torch.max(boxes1[:, :, 0][:, :, None], boxes2[:, :, 0][:, None])
    y1_max = torch.max(boxes1[:, :, 1][:, :, None], boxes2[:, :, 1][:, None])
    x2_min = torch.min(boxes1[:, :, 2][:, :, None], boxes2[:, :, 2][:, None])
    y2_min = torch.min(boxes1[:, :, 3][:, :, None], boxes2[:, :, 3][:, None])

    w = torch.clamp(x2_min - x1_max, min=0)
    h = torch.clamp(y2_min - y1_max, min=0)

    intersections = w * h
    return intersections

def non_overlap_loss(boxes):
    """
    计算边界框之间的非重叠损失。
    参数:
        boxes (Tensor): 形状为(batch_size, num_boxes, 4)的边界框张量。
    返回:
        loss (Tensor): 标量损失值。
    """
    batch_size, num_boxes, _ = boxes.size()
    # boxes = boxes.view(-1, 4)  # 展平为(batch_size * num_boxes, 4)

    intersections = box_intersection(boxes, boxes)

    # 计算每个边界框与其他边界框的交集面积之和
    intersection_sums = torch.sum(intersections, dim=2)

    # 计算边界框自身的面积
    # boxes_areas = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])

    # 计算损失
    loss = intersection_sums .mean()
    return loss

