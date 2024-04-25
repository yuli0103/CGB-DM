import sys

from PIL import Image
import torch
import cv2
from utils import logger
import os
import numpy as np
from utils.visual import draw_single
from module.model_diffusion import Diffusion
from transformers import set_seed
from utils.metric import metric
from torchvision import transforms
from utils.util import box_xyxy_to_cxcywh
from preprocess.find_bouding_box_visual import find_bounding_box

device = torch.device(f"cuda:{4}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
max_elem = 16

def get_detect_box(sal_path, width, height):
    img_cv = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
    detect_box = find_bounding_box(img_cv)
    detect_box = torch.tensor(detect_box)
    detect_box = box_xyxy_to_cxcywh(detect_box)
    detect_box = detect_box.float()
    detect_box[::2] /= width
    detect_box[1::2] /= height
    detect_box = detect_box.unsqueeze(0).unsqueeze(0).to(device)
    detect_box = 2 * (detect_box - 0.5)

    return detect_box

def main():
    set_seed(42)
    width = 513
    height = 750
    image_id = '86.png'
    image_path ='/mnt/data/kl23/pku/nosplit/train/inpainted_images/' + image_id
    sal_path = '/mnt/data/kl23/pku/nosplit/train/saliency/' + image_id
    sal_sub_path = '/mnt/data/kl23/pku/nosplit/train/saliency_sub/' + image_id
    checkpoint_path = '/mnt/data/kl23/checkpoint/pku/04_24_05/Epoch476_model_weights.pth'
    save_dir = '/mnt/data/kl23/result/pku/test_result/select_graph/x0/'
    save_inter = True

    model_ddpm = Diffusion(num_timesteps=1000, n_head=8, dim_transformer=512,
                           feature_dim=1024, seq_dim=8, num_layers=4,
                           device=device, ddim_num_steps=100, max_elem=max_elem,
                           beta_schedule='cosine')
    weights = torch.load(checkpoint_path ,map_location=device)
    model_ddpm.model.load_state_dict(weights)
    model_ddpm.model.eval()

    transform_rgb = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    transform_l = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
    img = Image.open(image_path).convert("RGB")
    img_sal = Image.open(sal_path).convert("L")
    img_sal_sub = Image.open(sal_sub_path).convert("L")

    img_sal_map = Image.fromarray(np.maximum(np.array(img_sal), np.array(img_sal_sub)))
    img_bg = transform_rgb(img)
    img_sal_map = transform_l(img_sal_map)
    img_tensor = torch.concat([img_bg, img_sal_map])
    img_tensor = img_tensor.unsqueeze(0).to(device)

    detect_box = get_detect_box(sal_path, width, height)

    if save_inter:
        save_inter_dir='/mnt/data/kl23/result/pku/test_result/select_graph/inter/'
        os.makedirs(save_inter_dir, exist_ok=True)
    else:
        save_inter_dir=None
    box, cls, mask = model_ddpm.reverse_ddim(img_tensor, detect_box,
                                             save_inter_dir=save_inter_dir,
                                             img=img,
                                             save_inter=save_inter,
                                             max_len=max_elem)

    draw_single(box, cls, img, 86, save_dir, width, height)

if __name__ == "__main__":
    main()