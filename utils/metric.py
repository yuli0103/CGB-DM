import sys
import torch
import os
import copy
import numpy as np
import cv2
from PIL import Image, ImageDraw
from math import log
from collections import OrderedDict
from einops import rearrange, reduce, repeat
from utils import logger
from utils.util import box_cxcywh_to_xyxy
'''
指标值因各个论文计算方法和代码不同，需要考虑很多
总体来说按如下分类
1. DS-GAN、LayoutPrompter：沿用的是DS-GAN的指标计算方法，主要是在PKU数据集上计算的，其中ali值可能计算有误
2. RDAM、CGL-GAN、PDA-GAN：沿用的是CGL-GAN的指标计算方法，主要是在CGL数据集上计算的，Rsub值不去计算，部分值的含义可以和DS-GAN对应
3. Ralf：开放了代码，并且有比较完善的的指标计算方法，在两个数据集均有计算，但部分指标计算方式和之前不同，有量级的差别
4. Design：计算不多，layout也不是主要任务，计算方式和Ralf类似
5. ICVT:数据集不同，指标也较差，无需比较
'''
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

def _extract_grad(image):
    image_npy = np.array(image * 255)
    image_npy_gray = cv2.cvtColor(image_npy, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(image_npy_gray, -1, 1, 0)
    grad_y = cv2.Sobel(image_npy_gray, -1, 0, 1)
    grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
    # ?: is it really OK to do content adaptive normalization?
    grad_xy = grad_xy / np.max(grad_xy)
    return torch.from_numpy(grad_xy)

def img_to_g_xy(img):
    img_cv_gs = np.uint8(cvt_pilcv(img, "pil2cv", cv2.COLOR_RGB2GRAY))
    # Sobel(src, ddepth, dx, dy)
    grad_x = cv2.Sobel(img_cv_gs, -1, 1, 0)
    grad_y = cv2.Sobel(img_cv_gs, -1, 0, 1)
    grad_xy = ((grad_x ** 2 + grad_y ** 2) / 2) ** 0.5
    grad_xy = grad_xy / np.max(grad_xy) * 255
    img_g_xy = Image.fromarray(grad_xy).convert('L')
    return img_g_xy


def metrics_iou(bb1, bb2):
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

    return a_inter / (a_1 + a_2 - a_inter)

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

    return a_inter / a_2

def is_contain(bb1, bb2):
    xl_1, yl_1, xr_1, yr_1 = bb1
    xl_2, yl_2, xr_2, yr_2 = bb2

    c1 = xl_1 <= xl_2
    c2 = yl_1 <= yl_2
    # c3 = xr_2 >= xr_2
    c3 = xr_1 >= xr_2
    c4 = yr_1 >= yr_2

    return c1 and c2 and c3 and c4

def ali_g(x):
    return -log(1 - x, 10)


def ali_delta(xs):
    n = len(xs)
    min_delta = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(xs[i] - xs[j])
            min_delta = min(min_delta, delta)
    return min_delta



def val_dsgan(img_size, clses, boxes):
    """
    The ratio of non-empty layouts.
    Higher is better.
    """
    w, h = img_size

    total_elem = 0
    empty_elem = 0

    for cls, box in zip(clses, boxes):
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        total_elem += len(mask_box)
        for mb in mask_box:
            xl, yl, xr, yr = mb
            xl = max(0, xl)
            yl = max(0, yl)
            xr = min(w, xr)
            yr = min(h, yr)
            if abs((xr - xl) * (yr - yl)) < (w * h / 1000):
                empty_elem += 1
    if total_elem:
        return 1 - empty_elem / total_elem
    else:
        return 0

def getRidOfInvalid(img_size, clses, boxes):
    w, h = img_size

    for i, (cls, box) in enumerate(zip(clses, boxes)):
        for j, b in enumerate(box):
            xl, yl, xr, yr = b
            xl = max(0, xl)
            yl = max(0, yl)
            xr = min(w, xr)
            yr = min(h, yr)
            if abs((xr - xl) * (yr - yl)) < (w * h / 1000):
                if clses[i, j]:
                    clses[i, j] = 0
    return clses

def ove_design(clses, boxes):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss
    mask = (clses != 0)
    mask = mask.squeeze(-1)
    bbox = boxes.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = bbox.unsqueeze(-1)
    l2, t2, r2, b2 = bbox.unsqueeze(-2)
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool,
                          device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = torch.nan_to_num(ai / a1)
    res = torch.nan_to_num(ar.sum(dim=(1, 2)) / mask.float().sum(-1))
    return res.mean()

def ali_design(img_size, clses, boxes):     # 以10为底
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss
    w, h = img_size
    bbox = boxes.permute(2, 0, 1)
    mask = (clses != 0)
    mask = mask.squeeze(-1)
    xl, yt, xr, yb = bbox
    xl, xr = xl / w, xr / w
    yt, yb = yt / h, yb / h
    xc, yc = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    # X = X.permute(0, 3, 2, 1)
    # X[~mask] = 1.
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.), 0.)

    X = -torch.log10(1 - X)
    res = torch.nan_to_num(X.sum(-1) / mask.float().sum(-1))
    return res.mean()

def ove_rdam(clses, boxes):
    overlap = []
    for cls, box in zip(clses, boxes):
        mask = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        mask_box = box[mask]
        n = len(mask_box)
        if n:
            for i in range(n):
                bb1 = mask_box[i]
                x1, y1, x2, y2 = bb1
                for j in range(i + 1, n):
                    bb2 = mask_box[j]
                    x3,y3,x4,y4 = bb2
                    x_over = max(min(x2, x4) - max(x1, x3), 0)
                    y_over = max(min(y2, y4) - max(y1, y3), 0)
                    overlap.append(x_over * y_over / ((x2-x1) * (y2-y1)) )
    return sum(overlap) / len(overlap) if len(overlap) != 0 else 0

def ali_rdam(img_size, clses, boxes):    # 以e为底
    R_ali = []
    w, h = img_size
    for cls, box in zip(clses, boxes):
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        if len(mask_box) <= 1:
           continue
        temp = []
        epsilon = 10e-7
        for i, lay in enumerate(mask_box):
            x1, y1, x2, y2 = mask_box[i]
            min_x = w
            min_y = h
            for j in range(0, len(mask_box)):
                if i == j:
                    continue
                x3, y3, x4, y4 = mask_box[j]
                min_x = min(abs(x1-x3), min_x, abs(x1+x2-x3-x4)/2, abs(x2-x4))
                min_y = min(abs(y1-y3), min_y, abs(y1+y2-y3-y4)/2, abs(y2-y4))
            min_xl = -np.log(1.0 - min_x / w  + epsilon)
            min_yl = -np.log(1.0 - min_y / h + epsilon)
            temp.append(min(min_xl, min_yl))

        if len(temp) != 0:
            R_ali.append(sum(temp) / len(temp))
    return sum(R_ali) / len(R_ali) if len(R_ali) else 0

def ove_dsgan(clses, boxes):
    """
    Ratio of overlapping area.
    Lower is better.
    """
    metrics = 0
    for cls, box in zip(clses, boxes):
        ove = 0
        mask = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        mask_box = box[mask]
        n = len(mask_box)
        cnt = 0
        if n:
            for i in range(n):
                bb1 = mask_box[i]
                for j in range(i + 1, n):
                    bb2 = mask_box[j]
                    ove += metrics_iou(bb1, bb2)
                    cnt += 1
            if cnt:
                metrics += ove / cnt
    return metrics / len(clses)

def ali_dsgan(img_size, clses, boxes):   # 以10为底
    """
    Indicator of the extent of non-alignment of pairs of elements.
    Lower is better.
    """
    w, h = img_size
    metrics = 0
    for cls, box in zip(clses, boxes):
        ali = 0
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        theda = []
        for mb in mask_box:
            pos = copy.deepcopy(mb)
            pos[0] /= w
            pos[2] /= w
            pos[1] /= h
            pos[3] /= h
            theda.append([pos[0], pos[1], (pos[0] + pos[2]) / 2, (pos[1] + pos[3]) / 2, pos[2], pos[3]])

        theda = [q.cpu().numpy() for t in theda for q in t]
        theda = np.array(theda)
        theda = theda.reshape(-1, 6)
        if theda.shape[0] <= 1:
            continue

        n = len(mask_box)
        for i in range(n):
            g_val = []
            for j in range(6):
                xys = theda[:, j]
                delta = ali_delta(xys)
                g_val.append(ali_g(delta))
            ali += min(g_val)
        metrics += ali

    return metrics / len(clses)

def und_l_dsgan(clses, boxes):
    """
    Overlap ratio of an underlay(deco) and a max-overlapped non-underlay(deco) element.
    Higher is better.
    """
    metrics = 0
    avali = 0
    for cls, box in zip(clses, boxes):
        und = 0
        mask_deco = (cls == 3).reshape(-1)
        mask_other = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        box_deco = box[mask_deco]
        box_other = box[mask_other]
        n1 = len(box_deco)
        n2 = len(box_other)
        if n1:
            avali += 1
            for i in range(n1):
                max_ios = 0
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    ios = metrics_inter_oneside(bb1, bb2)
                    max_ios = max(max_ios, ios)
                und += max_ios
            metrics += und / n1
    if avali > 0:
        return metrics / avali
    return 0

def und_s_dsgan(clses, boxes):
    """
    Overlap ratio of an underlay(deco) and a max-overlapped non-underlay(deco) element.
    Higher is better.
    """
    metrics = 0
    avali = 0
    for cls, box in zip(clses, boxes):
        und = 0
        mask_deco = (cls == 3).reshape(-1)
        mask_other = (cls > 0).reshape(-1) & (cls != 3).reshape(-1)
        box_deco = box[mask_deco]
        box_other = box[mask_other]
        n1 = len(box_deco)
        n2 = len(box_other)
        if n1:
            avali += 1
            for i in range(n1):
                bb1 = box_deco[i]
                for j in range(n2):
                    bb2 = box_other[j]
                    if is_contain(bb1, bb2):
                        und += 1
                        break
            metrics += und / n1
    if avali > 0:
        return metrics / avali
    return 0


def image_uti_dsgan_cgl(names, img_size, clses, boxes, input_dir, input_sub_dir):
    metrics = 0
    w, h= img_size

    for idx, name in enumerate(names):
        pic_1 = np.array(
            Image.open(os.path.join(input_dir,name.replace('.jpg','.png'))).convert("L").resize((w, h))) / 255
        pic_2 = np.array(
            Image.open(os.path.join(input_sub_dir, name.replace('.jpg', '.png'))).convert("L").resize((w, h))) / 255
        pic = np.maximum(pic_1, pic_2)

        c_pic = np.ones_like(pic) - pic

        cal_mask = np.zeros_like(pic)

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)

        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1

        total_not_sal = np.sum(c_pic)
        total_utils = np.sum(c_pic * cal_mask)

        if total_not_sal and total_utils:
            metrics += (total_utils / total_not_sal)
    return metrics / len(names)

def image_occ_dsgan_cgl(names, img_size, clses, boxes, input_dir, input_sub_dir):
    '''
    Average saliency of the pixels covered.
    Lower is better.
    '''
    metrics = 0
    w, h = img_size

    for idx, name in enumerate(names):
        pic_1 = np.array(
            Image.open(os.path.join(input_dir,name.replace('.jpg','.png'))).convert("L").resize((w, h))) / 255
        pic_2 = np.array(
            Image.open(os.path.join(input_sub_dir,name.replace('.jpg', '.png'))).convert("L").resize((w, h))) / 255
        pic = np.maximum(pic_1, pic_2)
        cal_mask = np.zeros_like(pic)

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)

        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]

        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1

        total_area = np.sum(cal_mask)
        total_sal = np.sum(pic[cal_mask == 1])
        if total_sal and total_area:
            metrics += (total_sal / total_area)
    return metrics / len(names)

def image_rea_dsgan_cgl(names, img_size, clses, boxes, input_dir):
    '''
    Average gradients of the pixels covered by predicted text-only elements.
    Lower is better.
    '''
    metrics = 0
    w, h = img_size
    for idx, name in enumerate(names):
        pic = Image.open(os.path.join(input_dir, name)).convert("RGB").resize((w, h))
        img_g_xy = np.array(img_to_g_xy(pic)) / 255
        cal_mask = np.zeros_like(img_g_xy)

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)

        text = (cls == 1).reshape(-1)
        text_box = box[text]
        deco = (cls == 3).reshape(-1)
        deco_box = box[deco]

        for mb in text_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        for mb in deco_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 0

        total_area = np.sum(cal_mask)
        total_grad = np.sum(img_g_xy[cal_mask == 1])
        if total_grad and total_area:
            metrics += (total_grad / total_area)
    return metrics / len(names)

def image_rea_ralf(names, img_size, clses, boxes, test_bg_dir):

    w, h = img_size
    results = []

    for idx, name in enumerate(names):
        rgb_image = np.array(Image.open(os.path.join(test_bg_dir, name)).convert("RGB").resize((w, h)))
        rgb_image = torch.from_numpy(np.array(rgb_image) / 255).float()

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
        # rea
        bbox_mask_special = torch.zeros(h, w)

        text = (cls == 1).reshape(-1)
        text_box = box[text]
        for mb in text_box:
            xl, yl, xr, yr = mb
            bbox_mask_special[yl:yr, xl:xr] = 1

        underlay = (cls == 3).reshape(-1)
        underlay_box = box[underlay]
        for mb in underlay_box:
            xl, yl, xr, yr = mb
            bbox_mask_special[yl:yr, xl:xr] = 0

        g_xy = _extract_grad(rgb_image)
        unreadability = g_xy[bbox_mask_special.bool()]
        if len(unreadability) == 0:
            results.append(0.0)
        else:
            results.append(unreadability.mean().item())

    rea = np.mean(np.array(results))
    return rea

from collections import defaultdict
def compute_saliency_aware_metrics(
    img_names, img_size, clses, boxes, test_bg_dir, test_sal_dir):
    """
    - utilization:
        Utilization rate of space suitable for arranging elements,
        Higher values are generally better (in 0.0 - 1.0 range).
    - occlusion:
        Average saliency of areas covered by elements.
        Lower values are generally better (in 0.0 - 1.0 range).
    - unreadability:
        Non-flatness of regions that text elements are solely put on
        Lower values are generally better.
    """
    w, h = img_size
    results = defaultdict(list)
    for idx, name in enumerate(img_names):
        saliency = np.array(Image.open(os.path.join(test_sal_dir, name.replace(".jpg",".png"))).convert("L").resize((w, h))) / 255
        saliency = torch.from_numpy(saliency).float()

        rgb_image = Image.open(os.path.join(test_bg_dir, name)).convert("RGB").resize((w, h))
        rgb_image = torch.from_numpy(np.array(rgb_image) / 255).float()
        inv_saliency = 1.0 - saliency

        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        cal_mask = torch.zeros_like(saliency)
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1

        # uti
        numerator = torch.sum(inv_saliency * cal_mask)
        denominator = torch.sum(inv_saliency)
        assert denominator > 0.0
        results["utilization"].append((numerator / denominator).item())
        # occ
        occlusion = saliency[cal_mask.bool()]
        if len(occlusion) == 0:
            results["occlusion"].append(0.0)
        else:
            results["occlusion"].append(occlusion.mean().item())
        # rea
        bbox_mask_special = torch.zeros_like(saliency)

        text = (cls == 1).reshape(-1)
        text_box = box[text]
        for mb in text_box:
            xl, yl, xr, yr = mb
            bbox_mask_special[yl:yr, xl:xr] = 1

        underlay = (cls == 3).reshape(-1)
        underlay_box = box[underlay]
        for mb in underlay_box:
            xl, yl, xr, yr = mb
            bbox_mask_special[yl:yr, xl:xr] = 0

        g_xy = _extract_grad(rgb_image)
        unreadability = g_xy[bbox_mask_special.bool()]
        if len(unreadability) == 0:
            results["unreadability"].append(0.0)
        else:
            results["unreadability"].append(unreadability.mean().item())
    uti = np.mean(np.array(results["utilization"]))
    occ = np.mean(np.array(results["occlusion"]))
    rea = np.mean(np.array(results["unreadability"]))

    return uti, occ, rea


def metric(names, model_output, test_bg_dir, test_sal_dir, test_sal_sub_dir, w=513, h=750):
    metric_res = {}
    clses = model_output[:, :, :1]
    boxes = model_output[:, :, 1:]

    boxes = box_cxcywh_to_xyxy(boxes)
    boxes = torch.clamp(boxes, 0, 1)
    boxes[:, :, ::2] = (boxes[:, :, ::2] * w).int()
    boxes[:, :, 1::2] = (boxes[:, :, 1::2] * h).int()
    metric_res['val'] = val_dsgan((w, h), clses, boxes)
    logger.log(f"val_dsgan:{metric_res['val']:.6f}")
    clses = getRidOfInvalid((w, h), clses, boxes)

    # ove_2 = ove_design(clses, boxes)
    # logger.log(f"ove_design:{ove_2:.6f}")
    # ali_2 = ali_dsgan((w, h), clses, boxes)
    # logger.log(f"ali_dsgan:{ali_2:.6f}")

    metric_res['ali'] = ali_design((w, h), clses, boxes)
    logger.log(f"ali_ralf:{metric_res['ali']:.6f}")
    metric_res['ove'] = ove_dsgan(clses, boxes)
    logger.log(f"ove_ralf:{metric_res['ove']:.6f}")

    # logger.log(f"ove_rdam:{ove_rdam(clses, boxes):.6f}")
    # logger.log(f"ali_rdam:{ali_rdam((w, h),clses, boxes):.6f}")
    # # logger.log(f"Rcom_rdam:{Rcom_rdam(names, clses, boxes)}")

    metric_res['undl'] = und_l_dsgan(clses, boxes)
    metric_res['unds'] = und_s_dsgan(clses, boxes)
    logger.log(f"und_l_ralf:{metric_res['undl']:.6f}")
    logger.log(f"und_s_ralf:{metric_res['unds']:.6f}")

    metric_res['occ'] = image_occ_dsgan_cgl(names, (w, h), clses, boxes, test_sal_dir, test_sal_sub_dir)
    logger.log(f"img_occ_ralf:{metric_res['occ']:.6f}")
    metric_res['uti'] = image_uti_dsgan_cgl(names, (w, h), clses, boxes, test_sal_dir, test_sal_sub_dir)
    logger.log(f"img_uti_ralf:{metric_res['uti']:.6f}")
    # rea = image_rea_dsgan_cgl(names, (w, h), clses, boxes, test_bg_dir)
    # logger.log(f"image_rea_dsgan:{rea:.6f}")

    # uti_ralf, occ_ralf, rea_ralf = compute_saliency_aware_metrics(names, (w, h), clses, boxes,
    #                                                               test_bg_dir, test_sal_dir_1)
    # logger.log(f"uti_ralf:{uti_ralf:.6f}, occ_ralf:{occ_ralf:.6f}, rea_ralf:{rea_ralf:.6f}")

    # logger.log(f"img_rea_ralf:{image_rea_ralf(names, (w, h), clses, boxes, test_bg_dir):.6f}")

    return metric_res