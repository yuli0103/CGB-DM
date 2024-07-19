import sys

from PIL import Image
import torch
from transformers import set_seed
from torchvision import transforms
import os
import numpy as np
import argparse

from utils.visualize import draw_image
from cgbdm.diffusion import Diffusion
from utils.util import box_xyxy_to_cxcywh, finalize, load_config
from data_process.generate_salbox_csv import find_bounding_box
from data_process.saliency_detection import get_saliency_map


def get_sal_box(img_sal_map, width, height, device):
    sal_box = find_bounding_box(img_sal_map, ifpath=False)
    sal_box = box_xyxy_to_cxcywh(torch.tensor(sal_box)).float()
    sal_box[::2] /= width
    sal_box[1::2] /= height
    sal_box = sal_box.unsqueeze(0).unsqueeze(0).to(device)
    sal_box = 2 * (sal_box - 0.5)

    return sal_box

def main(opt):
    seed = 1
    set_seed(seed)

    device = torch.device(f"cuda:{opt.gpuid}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    cfg = load_config(f'configs/{opt.dataset}_anno_test.yaml')
    
    img_inp = Image.open(opt.image_path).convert("RGB")
    cfg.width, cfg.height = img_inp.size
    # get saliency map

    img_sal_map = get_saliency_map(img_inp)
    # get saliency bounding box
    sal_box = get_sal_box(img_sal_map, cfg.width, cfg.height, device)

    transform_rgb = transforms.Compose([
        transforms.Resize([384, 256]),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    transform_l = transforms.Compose([
        transforms.Resize([384, 256]),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5], inplace=True)
    ])
    img_tensor = torch.concat([transform_rgb(img_inp), transform_l(img_sal_map)])
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    diffusion_model = Diffusion(num_timesteps=1000,
                                ddim_num_steps=100,
                                n_head=cfg.n_head,
                                dim_model=cfg.d_model,
                                feature_dim=cfg.feature_dim,
                                seq_dim=cfg.num_class + 4,
                                num_layers=cfg.n_layers,
                                device=device,
                                max_elem=cfg.max_elem,)

    weights = torch.load(opt.check_path ,map_location=device)
    diffusion_model.model.load_state_dict(weights)
    diffusion_model.model.eval()

    bbox, cls, _ = diffusion_model.reverse_ddim(img_tensor, sal_box, cfg, save_inter=False)

    current_dir = os.getcwd()
    imgname = os.path.splitext(os.path.split(opt.image_path))
    img_name = 'render_' + imgname
    draw_image(bbox, cls, img_inp, img_name, cfg.width, cfg.height, cfg.numclass, save_dir=current_dir)

# Start with main code
if __name__ == "__main__":
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpuid',
        type=int,
        default=0,
        help='choose gpu')
    parser.add_argument(
        '--render_style',
        type=str,
        default='pku',
        help='(pku, cgl)')
    parser.add_argument(
        '--image_path',
        type=str,
        default='',
        help='choose image'
    )
    parser.add_argument(
        '--check_path',
        type=str,
        default='',
        help='choose checkpoint'
    )
    opt = parser.parse_args()
    main(opt)