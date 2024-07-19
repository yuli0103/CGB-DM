import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os
from einops import rearrange, reduce, repeat
from math import log
import copy
from collections import defaultdict
import cv2

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

def _extract_grad(image):
    image_npy = np.array(image * 255)
    image_npy_gray = cv2.cvtColor(image_npy, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(image_npy_gray, -1, 1, 0)
    grad_y = cv2.Sobel(image_npy_gray, -1, 0, 1)
    grad_xy = ((grad_x**2 + grad_y**2) / 2) ** 0.5
    # ?: is it really OK to do content adaptive normalization?
    grad_xy = grad_xy / np.max(grad_xy)
    return torch.from_numpy(grad_xy)

'''RADM:Relation-Aware Diffusion Model for Controllable Poster Layout Generation'''
def Rea_radm(img_names, clses, boxes):
    def nn_conv2d(im, sobel_kernel):
        conv_op = nn.Conv2d(1, 1, 3, bias=False)
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        conv_op.weight.data = torch.from_numpy(sobel_kernel)
        gradient = conv_op(Variable(im))
        return gradient

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype='float32')
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype='float32')

    R_coms = []
    for idx, name in enumerate(img_names):
        gray = Image.open(os.path.join("Dataset/test/image_canvas", name)).convert("RGB").resize((513, 750))
        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
        for c, b in zip(cls, box):
            if c == 1:
                x1, y1, x2, y2 = b
                gray = gray.crop((x1, y1, x2, y2))
                gray_array = np.array(gray, dtype='float32')
                gray_tensor = torch.from_numpy(gray_array.reshape((1, 1, gray_array.shape[0], gray_array.shape[1])))

                try:  # The w or h can be 0
                    image_x = nn_conv2d(gray_tensor, sobel_x)
                    image_y = nn_conv2d(gray_tensor, sobel_y)
                    image_xy = torch.mean(torch.sqrt(image_x ** 2 + image_y ** 2)).detach().numpy()
                    R_coms.append(image_xy)
                except:
                    continue

    return sum(R_coms) / len(R_coms) if len(R_coms) != 0 else 0

def Overlap_rdam(clses, boxes):
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

def Alignment_rdam(img_size, clses, boxes):    # 以e为底
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

'''DS-GAN:PosterLayout: A New Benchmark and Approach for Content-aware Visual-Textual Presentation Layout'''
def Validity_dsgan(clses, boxes):
    """
    The ratio of non-empty layouts.
    Higher is better.
    """
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
            xr = min(1, xr)
            yr = min(1, yr)
            if abs((xr - xl) * (yr - yl)) < (1 / 1000):
                empty_elem += 1
    if total_elem:
        return 1 - empty_elem / total_elem
    else:
        return 0

def Alignment_dsgan(img_size, clses, boxes):   # 以10为底
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

'''Desigen: A Pipeline for Controllable Design Template Generation'''
def Overlap_design(clses, boxes):
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

def Alignment_design(img_size, clses, boxes):     # 以10为底
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

'''Calculation of indicators based on .pt files'''
def Utilization_basedpt(occ_matrix, clses, boxes):
    results = []
    B = occ_matrix.shape[0]
    for idx in range(B):
        saliency = occ_matrix[idx]
        inv_saliency = 1.0 - saliency
        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        cal_mask = torch.zeros_like(saliency)
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        numerator = torch.sum(inv_saliency * cal_mask)
        denominator = torch.sum(inv_saliency)
        assert denominator > 0.0
        results.append((numerator / denominator).item())
    uti = np.mean(np.array(results))
    return uti

def Occlusion_basedpt(occ_matrix, clses, boxes):
    '''
    Average saliency of the pixels covered.
    Lower is better.
    '''
    results = []
    B = occ_matrix.shape[0]
    for idx in range(B):
        saliency = occ_matrix[idx]
        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
        mask = (cls > 0).reshape(-1)
        mask_box = box[mask]
        cal_mask = torch.zeros_like(saliency)
        for mb in mask_box:
            xl, yl, xr, yr = mb
            cal_mask[yl:yr, xl:xr] = 1
        occlusion = saliency[cal_mask.bool()]
        if len(occlusion) != 0:
            results.append(occlusion.mean().item())
    occ = np.mean(np.array(results))
    return occ

def Unreadability_basedpt(rea_matrix,img_size, clses, boxes):
    w, h = img_size
    results = []
    B = clses.shape[0]
    for idx in range(B):
        cls = np.array(clses[idx].cpu(), dtype=int)
        box = np.array(boxes[idx].cpu(), dtype=int)
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
        g_xy = rea_matrix[idx].squeeze(0)
        unreadability = g_xy[bbox_mask_special.bool()]

        if len(unreadability) == 0:
            results.append(0.0)
        else:
            results.append(unreadability.mean().item())
    rea = np.mean(np.array(results))
    return rea

'''RALF:Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation'''
def Alignment_ralf(img_size, clses, boxes):
    """
    Computes some alignment metrics that are different to each other in previous works.
    Lower values are generally better.
    Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    """
    w, h = img_size
    bbox = boxes.permute(2, 0, 1)
    mask = (clses != 0)
    mask = mask.squeeze(-1)
    _, S = mask.size()

    xl, yt, xr, yb = bbox
    xl, xr = xl / w, xr / w
    yt, yb = yt / h, yb / h
    xc, yc = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)
    X = -torch.log10(1 - X)

    # original
    # return X.sum(-1) / mask.float().sum(-1)

    score = reduce(X, "b s -> b", reduction="sum")
    score_normalized = score / reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    Y = torch.stack([xl, xc, xr], dim=1)
    Y = rearrange(Y, "b x s -> b x 1 s") - rearrange(Y, "b x s -> b x s 1")

    batch_mask = rearrange(~mask, "b s -> b 1 s") | rearrange(~mask, "b s -> b s 1")
    idx = torch.arange(S, device=Y.device)
    batch_mask[:, idx, idx] = True
    batch_mask = repeat(batch_mask, "b s1 s2 -> b x s1 s2", x=3)
    Y[batch_mask] = 1.0

    Y = reduce(Y.abs(), "b x s1 s2 -> b s1", "min")
    Y[Y == 1.0] = 0.0
    score_Y = reduce(Y, "b s -> b", "sum")

    results = {
        "alignment-ACLayoutGAN": score.mean(),  # Because it may be confusing.
        "alignment-LayoutGAN++": score_normalized.mean(),
        "alignment-NDN": score_Y.mean(),  # Because it may be confusing.
    }
    # return {k: v.tolist() for (k, v) in results.items()}
    return results["alignment-LayoutGAN++"]

def Content_aware_metrics_ralf(
    img_names, img_size, clses, boxes, test_inp_dir, test_sal_dir, test_sal_sub_dir):
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
        pic_1 = np.array(Image.open(os.path.join(test_sal_dir, name)).convert("L").resize((240,350))) / 255
        pic_2 = np.array(
            Image.open(os.path.join(test_sal_sub_dir, name)).convert("L").resize((240,350))) / 255
        pic = np.maximum(pic_1, pic_2)
        saliency = torch.from_numpy(pic).float()

        rgb_image = Image.open(os.path.join(test_inp_dir, name)).convert("RGB").resize((240,350))
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