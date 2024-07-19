import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
import os
from einops import rearrange, reduce, repeat

device = "cuda" if torch.cuda.is_available() else "cpu"

def Rcom_radm(img_names, clses, boxes):
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
        gray = Image.open(os.path.join("Dataset/test/image_canvas", name)).convert("L").resize((513, 750))
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