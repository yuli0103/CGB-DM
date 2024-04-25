import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
import os
from utils.util import box_cxcywh_to_xyxy
import numpy as np
import seaborn as sns

def get_colors(n_colors: int) -> list[tuple[int, int, int]]:
    colors = sns.color_palette("husl", n_colors=n_colors)
    colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
    return colors  # type: ignore

def draw_box_pku(img, elems, elems2):
    # drawn_outline = img.copy()
    drawn_fill = img.copy()
    # draw_ol = ImageDraw.Draw(drawn_outline, "RGBA")
    # draw_f = ImageDraw.ImageDraw(drawn_fill)
    draw_f = ImageDraw.Draw(drawn_fill, "RGBA")
    # colors = {1: 'green', 2: 'red', 3: 'orange'}
    # colors = {1: '#91CD7E', 2: '#E291A2', 3: '#6BB3E3'}
    colors = get_colors(3)
    colors[0], colors[1] = colors[1], colors[0]
    # for cls, box in elems:
    #     if cls:
    #         draw_ol.rectangle(tuple(box), fill=None, outline=colors[int(cls)], width=2)

    s_elems = sorted(list(elems2), key=lambda x: x[0], reverse=True)
    for cls, box in s_elems:
        if cls:
            # c_fill = colors[int(cls)]
            label = int(cls) - 1
            c_fill = colors[label] + (160,)
            draw_f.rectangle(tuple(box), outline=colors[label], fill=c_fill)

    # drawn_outline = drawn_outline.convert("RGBA")
    # drawn_fill = drawn_fill.convert("RGBA")
    # drawn_fill.putalpha(int(256 * 0.9))
    # drawn = Image.alpha_composite(drawn_outline, drawn_fill)

    return drawn_fill


def draw_box_cgl(img, elems, elems2):
    drawn_outline = img.copy()
    drawn_fill = img.copy()
    draw_ol = ImageDraw.ImageDraw(drawn_outline)
    draw_f = ImageDraw.ImageDraw(drawn_fill)
    cls_color_dict = {1: 'green', 2: 'red', 3: 'orange', 4: 'pink'}

    for cls, box in elems:
        if cls:
            draw_ol.rectangle(tuple(box), fill=None, outline=cls_color_dict[cls], width=5)

    s_elems = sorted(list(elems2), key=lambda x: x[0], reverse=True)
    for cls, box in s_elems:
        if cls:
            draw_f.rectangle(tuple(box), fill=cls_color_dict[cls])

    drawn_outline = drawn_outline.convert("RGBA")
    drawn_fill = drawn_fill.convert("RGBA")
    drawn_fill.putalpha(int(256 * 0.3))
    drawn = Image.alpha_composite(drawn_outline, drawn_fill)

    return drawn

# def draw_box_f(img, cls_list, box_list):
#     img_copy = img.copy()
#     draw = ImageDraw.ImageDraw(img_copy)
#     cls_color_dict = {0: "black", 1: 'green', 2: 'red', 3: 'orange'}
#     for cls, box in zip(cls_list, box_list):
#         if cls:
#             draw.rectangle(box, fill=None, outline=cls_color_dict[int(cls)], width=5)
#     return img_copy

def draw(box, cls, imgs, id, dir, w, h):
    plt.figure(figsize=(12, 5))

    for idx, (c, b) in enumerate(zip(cls, box)):

        c = c.detach().cpu().numpy()
        b = box_cxcywh_to_xyxy(b.detach().cpu())
        b = torch.clamp(b, 0, 1)
        b[:, ::2] = (b[:, ::2] * w).int()
        b[:, 1::2] = (b[:, 1::2] * h).int()
        b = b.numpy()
        img = imgs[idx]
        # drawn = draw_box_f(img, c.numpy(), b.numpy())
        drawn = draw_box_pku(img, zip(c, b), zip(c, b))

        plt.subplot(1, 4, idx + 1)
        plt.axis("off")
        plt.imshow(drawn)

    dir = dir + '{}.png'.format(id)
    print(f"save:{id}")
    plt.savefig(dir)
    plt.close()
    return

def draw_single(box, cls, img, id, dir, w, h):
    c = cls.detach().cpu().numpy()
    if c.ndim>=2:
        c = c.squeeze(0)
    if box.ndim>=3:
        box = box.squeeze(0)
    b = box_cxcywh_to_xyxy(box.detach().cpu())
    b = torch.clamp(b, 0, 1)
    b[:, ::2] = (b[:, ::2] * w).int()
    b[:, 1::2] = (b[:, 1::2] * h).int()
    b = b.numpy()
    drawn = draw_box_pku(img, zip(c, b), zip(c, b))
    dir = dir + '{}.png'.format(id)
    drawn.save(os.path.join(dir))