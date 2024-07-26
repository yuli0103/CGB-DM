import sys

import torch
import os
import copy
import numpy as np
import cv2
from PIL import Image, ImageDraw
from math import log
from einops import rearrange, reduce, repeat
from utils import logger
from utils.util import box_cxcywh_to_xyxy
from torch import Tensor
from typing import Callable, Optional, Union, Any
from torchvision.transforms.functional import to_tensor

def _mean(values: list[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    else:
        return sum(values) / len(values)

def cvt_pilcv(img, req='pil2cv', color_code=None):
    if req == 'pil2cv':
        if color_code == None:
            color_code = cv2.COLOR_RGB2BGR
        dst = cv2.cvtColor(np.asarray(img), color_code)
    elif req == 'cv2pil':
        if color_code == None:
            color_code = cv2.COLOR_BGR2RGB
        dst = Image.fromarray(cv2.cvtColor(img, color_code))
    return dst

def img_to_g_xy(img):
    img_cv_gs = np.uint8(cvt_pilcv(img, "pil2cv", cv2.COLOR_RGB2GRAY))
    # Sobel(src, ddepth, dx, dy)
    grad_x = cv2.Sobel(img_cv_gs, -1, 1, 0)
    grad_y = cv2.Sobel(img_cv_gs, -1, 0, 1)
    grad_xy = ((grad_x ** 2 + grad_y ** 2) / 2) ** 0.5
    grad_xy = grad_xy / np.max(grad_xy) * 255
    img_g_xy = Image.fromarray(grad_xy).convert('L')
    return img_g_xy

def _extract_grad(image):
    image_npy = np.array(image * 255)
    image_npy_gray = cv2.cvtColor(image_npy, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(image_npy_gray, -1, 1, 0)
    grad_y = cv2.Sobel(image_npy_gray, -1, 0, 1)
    grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
    # ?: is it really OK to do content adaptive normalization?
    grad_xy = grad_xy / np.max(grad_xy)
    return torch.from_numpy(grad_xy)


def _list_all_pair_indices(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate all pairs
    """
    N = bbox.shape[0]
    ii, jj = np.meshgrid(range(N), range(N))
    ii, jj = ii.flatten(), jj.flatten()
    is_non_diag = ii != jj  # IoU for diag is always 1.0
    ii, jj = ii[is_non_diag], jj[is_non_diag]
    return ii, jj


def metrics_inter_oneside(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2

    w_1 = xr_1 - xl_1
    w_2 = xr_2 - xl_2
    h_1 = yr_1 - yl_1
    h_2 = yr_2 - yl_2

    w_inter = min(xr_1, xr_2) - max(xl_1, xl_2)
    h_inter = min(yr_1, yr_2) - max(yl_1, yl_2)

    a_1 = w_1 * h_1
    a_2 = w_2 * h_2
    a_inter = w_inter * h_inter
    if w_inter <= 0 or h_inter <= 0:
        a_inter = 0

    l_max = np.maximum(xl_1, xl_2)
    r_min = np.minimum(xr_1, xr_2)
    t_max = np.maximum(yl_1, yr_1)
    b_min = np.minimum(yl_2, yr_2)
    cond = (l_max < r_min) & (t_max < b_min)

    a_i = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a_1))

    return a_inter / a_2

def _compute_iou_group(
    box_1: Union[np.ndarray, Tensor],
    box_2: Union[np.ndarray, Tensor],
    method: str = "iou",
    transform: bool = True,
) -> np.ndarray:
    assert method in ["iou", "giou", "ai/a1", "ai/a2"]

    if isinstance(box_1, Tensor):
        box_1 = np.array(box_1)
        box_2 = np.array(box_2)
    assert len(box_1) == len(box_2)

    if transform:
        l1, t1, r1, b1 = box_1.T
        l2, t2, r2, b2 = box_2.T
    else:
        l1, t1, r1, b1 = box_1
        l2, t2, r2, b2 = box_2
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = np.maximum(l1, l2)
    r_min = np.minimum(r1, r2)
    t_max = np.maximum(t1, t2)
    b_min = np.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    if transform:
        ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1[0]))
    else:
        ai = np.where(cond, (r_min - l_max) * (b_min - t_max), np.zeros_like(a1))

    au = a1 + a2 - ai
    iou = ai / au

    if method == "iou":
        return iou
    elif method == "ai/a1":
        return ai / a1
    elif method == "ai/a2":
        return ai / a2

    # outer region
    l_min = np.minimum(l1, l2)
    r_max = np.maximum(r1, r2)
    t_min = np.minimum(t1, t2)
    b_max = np.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou: np.ndarray = iou - (ac - au) / ac

    return giou

def is_contain(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2
    c1 = xl_1 <= xl_2
    c2 = yl_1 <= yl_2
    c3 = xr_1 >= xr_2
    c4 = yr_1 >= yr_2
    return c1 and c2 and c3 and c4


def validity_cal(clses, boxes):
    mask = clses > 0
    valid_boxes = boxes[mask.squeeze(-1)]

    if valid_boxes.numel() == 0:
        return 0

    clamped_boxes = torch.clamp(valid_boxes, 0, 1)
    areas = (clamped_boxes[:, 2] - clamped_boxes[:, 0]) * (clamped_boxes[:, 3] - clamped_boxes[:, 1])

    empty_count = torch.sum(areas < 1e-3)
    return 1 - empty_count.float() / valid_boxes.shape[0]

# def validity_cal(clses, boxes):
#     total_elem = 0
#     empty_elem = 0
#     for cls, box in zip(clses, boxes):
#         mask = (cls > 0).reshape(-1)
#         mask_box = box[mask]
#         total_elem += len(mask_box)
#         for mb in mask_box:
#             xl, yl, xr, yr = mb
#             xl = max(0, xl)
#             yl = max(0, yl)
#             xr = min(1, xr)
#             yr = min(1, yr)
#             if abs((xr - xl) * (yr - yl)) < (1 / 1000):
#                 empty_elem += 1
#     if total_elem:
#         return 1 - empty_elem / total_elem
#     else:
#         return 0

def getRidOfInvalid(clses, boxes):
    clamped_boxes = torch.clamp(boxes, 0, 1)
    areas = (clamped_boxes[..., 2] - clamped_boxes[..., 0]) * (clamped_boxes[..., 3] - clamped_boxes[..., 1])
    invalid_mask = areas < 1e-3
    clses[invalid_mask] = 0
    return clses
# def getRidOfInvalid(clses, boxes):
#     for i, (cls, box) in enumerate(zip(clses, boxes)):
#         for j, b in enumerate(box):
#             xl, yl, xr, yr = b
#             xl = max(0, xl)
#             yl = max(0, yl)
#             xr = min(1, xr)
#             yr = min(1, yr)
#             if abs((xr - xl) * (yr - yl)) < (1 / 1000):
#                 if clses[i, j]:
#                     clses[i, j] = 0
#     return clses


def overlap_cal(clses, boxes):
    """
    Ratio of overlapping area.
    Lower is better.
    """
    metrics = []
    for cls, box in zip(clses, boxes):
        mask = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        mask_box = box[mask]
        n = len(mask_box)
        if n in [0, 1]:
            continue
        ii, jj = _list_all_pair_indices(mask_box)
        iou: np.ndarray = _compute_iou_group(mask_box[ii], mask_box[jj], method="iou", transform=True)
        result: float = iou.mean().item()
        metrics.append(result)
    return np.mean(np.array(metrics))


def underlay_cal(clses, boxes):
    """
    Overlap ratio of an underlay(deco) and a max-overlapped non-underlay(deco) element.
    Higher is better.
    """
    metric_l = []
    metric_s = []
    thresh = 1.0 - np.finfo(np.float32).eps

    for cls, box in zip(clses, boxes):
        mask_und = (cls == 3).reshape(-1)
        mask_other = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        box_und = box[mask_und]
        box_other = box[mask_other]
        n1 = len(box_und)
        n2 = len(box_other)
        if n1:
            for i in range(n1):
                max_iou = 0
                bb1 = box_und[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    # ios = metrics_inter_oneside(bb1, bb2)
                    ios = _compute_iou_group(bb1, bb2, method="ai/a2", transform=False)
                    max_iou = max(max_iou, ios)
                strict_score = (max_iou >= thresh).any().astype(np.float32)
                metric_l.append(max_iou)
                metric_s.append(strict_score)

    return np.mean(np.array(metric_l)), np.mean(np.array(metric_s))


def utilization_cal(img_names, clses, boxes, cfg):
    metric = 0
    img_size = (cfg.width, cfg.height)
    for idx, name in enumerate(img_names):
        sal_1 = np.array(Image.open(os.path.join(cfg.paths.test.sal_dir, name)).convert("L"))
        sal_2 = np.array(Image.open(os.path.join(cfg.paths.test.sal_sub_dir, name)).convert("L"))
        sal_map = Image.fromarray(np.maximum(sal_1, sal_2))
        sal_map = to_tensor(sal_map.resize(img_size))
        sal_map = rearrange(sal_map, "1 h w ->h w")
        inv_saliency = 1.0 - sal_map

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        cal_mask = torch.zeros_like(sal_map)
        # cal_mask[mask_box[:, 1]:mask_box[:, 3], mask_box[:, 0]:mask_box[:, 2]] = True
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1

        numerator = torch.sum(inv_saliency * cal_mask)
        denominator = torch.sum(inv_saliency)
        assert denominator > 0.0
        metric += (numerator / denominator).item()
    return metric / len(img_names)

def occlusion_cal(img_names, clses, boxes, cfg):
    '''
    Average saliency of the pixels covered.
    Lower is better.
    '''
    metric = 0
    img_size = (cfg.width, cfg.height)

    for idx, name in enumerate(img_names):
        sal_1 = np.array(Image.open(os.path.join(cfg.paths.test.sal_dir, name)).convert("L"))
        sal_2 = np.array(Image.open(os.path.join(cfg.paths.test.sal_sub_dir, name)).convert("L"))
        sal_map = Image.fromarray(np.maximum(sal_1, sal_2))
        sal_map = to_tensor(sal_map.resize(img_size))
        sal_map = rearrange(sal_map, "1 h w ->h w")

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)

        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        cal_mask = torch.zeros_like(sal_map)

        # cal_mask[mask_box[:, 1]:mask_box[:, 3], mask_box[:, 0]:mask_box[:, 2]] = True
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        occlusion = sal_map[cal_mask.bool()]
        if len(occlusion) != 0:
            metric += occlusion.mean().item()

    return metric / len(img_names)

def unreadability_cal(img_names, clses, boxes, cfg):
    metrics = []
    img_size = (cfg.width, cfg.height)

    for idx, name in enumerate(img_names):
        image = to_tensor(Image.open(os.path.join(cfg.paths.test.inp_dir, name)).convert("RGB").resize(img_size))
        image = rearrange(image, "c h w ->h w c")

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)

        bbox_mask_special = torch.zeros(cfg.height, cfg.width)
        text_mask = (cls == 1).reshape(-1)
        text_boxes = box[text_mask]
        # if text_boxes.numel() > 0:
            # bbox_mask_special[text_boxes[:, 1]:text_boxes[:, 3], text_boxes[:, 0]:text_boxes[:, 2]] = True
        for mb in text_boxes:
            xl, yl, xr, yr = mb
            bbox_mask_special[yl:yr, xl:xr] = 1
        underlay_mask = (cls == 3).reshape(-1)
        underlay_boxes = box[underlay_mask]
        # if underlay_boxes.numel() > 0:
        #     bbox_mask_special[underlay_boxes[:, 1]:underlay_boxes[:, 3],
        #     underlay_boxes[:, 0]:underlay_boxes[:, 2]] = False
        for mb in underlay_boxes:
            xl, yl, xr, yr = mb
            bbox_mask_special[yl:yr, xl:xr] = 0
        g_xy = _extract_grad(image)
        unreadability = g_xy[bbox_mask_special.bool()]

        metrics.append(unreadability.mean().item() if unreadability.numel() > 0 else 0.0)

    return np.mean(np.array(metrics))

def metric(img_names, test_output, cfg):
    logger.log("Calculating metrics...")
    clses, boxes = test_output[:, :, :1], test_output[:, :, 1:]
    boxes = torch.clamp(box_cxcywh_to_xyxy(boxes), 0, 1)

    metrics = {
        'val': validity_cal(clses, boxes),
    }
    clses = getRidOfInvalid(clses, boxes)
    metrics['ove'] = overlap_cal(clses, boxes)
    metrics['undl'], metrics['unds'] = underlay_cal(clses, boxes)

    boxes[:, :, ::2] *= cfg.width
    boxes[:, :, 1::2] *= cfg.height
    boxes = boxes.round().int()

    for name, func in [
        ('occ', occlusion_cal),
        # ('uti', utilization_cal),
        ('rea', unreadability_cal)
    ]:
        metrics[name] = func(img_names, clses, boxes, cfg)

    for key, value in metrics.items():
        logger.log(f"{key}:{value:.6f}")

    return metrics
