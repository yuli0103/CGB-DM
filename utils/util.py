import sys

import torch
import numpy as np
import random
import os
import re
import fsspec
from torchvision.ops.boxes import box_area
from collections import OrderedDict
from typing import Callable, Optional, Union, Any
from torch import Tensor
import yaml
from datetime import datetime
import pytz

class Config:
    def __init__(self, config):
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def process_paths(config):
    base = config['paths']['base']

    sections = ['test']
    if 'train' in config['paths']:
        sections.append('train')

    for section in sections:
        if section in config['paths']:
            for key, value in config['paths'][section].items():
                config['paths'][section][key] = os.path.join(base, value)

    return config

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    config = process_paths(config)

    china_tz = pytz.timezone('Asia/Shanghai')
    config['datetime'] = datetime.now(china_tz).strftime('%m_%d_%H%M')
    return Config(config)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0) , (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - w/2), (y_c - h/2),
         (x_c + w/2), (y_c + h/2)]
    return torch.stack(b, dim=-1)

def convert_xywh_to_ltrb(
    bbox: Union[Tensor, np.ndarray, list[float]]
) -> Union[list[Tensor], list[np.ndarray], list[float]]:
    # assert len(bbox) == 4
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

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

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

def finalize(layout, num_class):
    bbox = layout[:, :, num_class:]
    bbox = torch.clamp(bbox, min=-1, max=1) / 2 + 0.5
    label = torch.argmax(layout[:, :, :num_class], dim=2)
    mask = (label != 0).clone().detach()
    label = label.unsqueeze(-1)
    return bbox, label, mask

def clamp_w_tol(
    value: float, tolerance: float = 5e-3, vmin: float = 0.0, vmax: float = 1.0
) -> float:
    """
    Clamp the value to [vmin, vmax] range with tolerance.
    """
    assert vmin - tolerance <= value <= vmax + tolerance, value
    return max(vmin, min(vmax, value))

def _compare(low: float, high: float) -> tuple[float, float]:
    if low > high:
        return high, low
    else:
        return low, high

def has_valid_area(width, height, thresh: float = 1e-3) -> bool:
    """
    Check whether the area is smaller than the threshold.
    """
    area = width * height
    valid = area > thresh
    return valid

def natural_sort_cmp(a, b):
    a_match = re.match(r'(\d+)\.(png|jpg)$', a, re.IGNORECASE)
    b_match = re.match(r'(\d+)\.(png|jpg)$', b, re.IGNORECASE)
    if a_match and b_match:
        a_num = int(a_match.group(1))
        b_num = int(b_match.group(1))
        return a_num - b_num
    elif a_match:
        return -1
    elif b_match:
        return 1
    else:
        return 0
