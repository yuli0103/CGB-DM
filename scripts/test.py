import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import logger
from torch.utils.data import DataLoader
from transformers import set_seed
from utils.metric import metric
from utils.util import finalize, load_config
from utils.visualize import visualize_images

import os
import argparse
from data_process.dataloader import test_uncond_dataset, test_cond_dataset
from cgbdm.diffusion import Diffusion

sys.path.append('/home/ly24/code/ditl')

def sample_uncond(diffusion_model, testing_dl, cfg):
    sample_output = []
    device = diffusion_model.device
    cnt = 0
    for idx, (image, sal_box) in enumerate(testing_dl):
        image, sal_box = image.to(device), sal_box.to(device)
        bbox, cls, _ = diffusion_model.reverse_ddim(image, sal_box, cfg, save_inter=False)
        samples = torch.cat([cls, bbox], dim=2)
        sample_output.append(samples.cpu())
        cnt = cnt + image.shape[0]
        logger.log(f"created {cnt} samples")

    sample_output = torch.concat(sample_output, dim=0)
    return sample_output
def sample_cond(diffusion_model, testing_dl, cfg, cond='c'):
    sample_output = []
    device = diffusion_model.device
    cnt = 0
    for idx, (image, layout, sal_box) in enumerate(testing_dl):
        image, layout, sal_box= image.to(device), layout.to(device), sal_box.to(device)
        box, cls, mask = diffusion_model.conditional_reverse_ddim(layout, image, sal_box, cfg, cond=cond)
        samples = torch.cat([cls, box], dim=2)
        sample_output.append(samples.cpu())

        cnt = cnt + image.shape[0]
        logger.log(f"created {cnt} samples")
 
    sample_output = torch.concat(sample_output, dim=0)
    return sample_output

def sample_refine(diffusion_model, testing_dl, cfg):
    samples = {'output': [], 'noise': [], 'gt': []}
    num_class = cfg.num_class
    cnt = 0
    device = diffusion_model.device

    for idx, (image, layout, sal_box) in enumerate(testing_dl):
        image, layout, sal_box = image.to(device), layout.to(device), sal_box.to(device)
        real_label = layout[:,:,:num_class]
        box_gt, cls_gt, mask_gt = finalize(layout, num_class)
        cls_gt[:,1:,:] = 0

        noise = torch.normal(0, 0.01, size=box_gt.size()).to(device)
        box_noise = torch.clamp(box_gt + noise, min=0, max=1)
        noise_layout = torch.cat((real_label, 2 * (box_noise - 0.5)), dim=2).to(device)

        box, cls, _ = diffusion_model.refinement_reverse_ddim(noise_layout, image, sal_box)

        for key, value in zip(samples.keys(), [
            torch.cat([cls, box], dim=2),
            torch.cat([cls_gt, box_noise], dim=2),
            torch.cat([cls_gt, box_gt], dim=2)
        ]):
            samples[key].append(value.cpu())

        cnt = cnt + image.shape[0]
        logger.log(f"created {cnt} samples")

    return [torch.cat(samples[key], dim=0) for key in samples.keys()]

def main(opt):
    seed = 1
    set_seed(seed)

    device = torch.device(f"cuda:{opt.gpuid}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    cfg = load_config(f'configs/{opt.dataset}_{opt.anno}_test.yaml')
    cfg.task = opt.task
    cfg.imgname_order_dir = os.path.join(cfg.imgname_order_dir, f'seed_{seed}_{opt.dataset}_{opt.anno}_test.pt')

    if cfg.task == 'uncond':
        testing_set = test_uncond_dataset(cfg)
    else:
        testing_set = test_cond_dataset(cfg)
    testing_dl = DataLoader(testing_set, num_workers=cfg.num_workers, batch_size=cfg.batch_size, shuffle=True)
    logger.log(f"Testing set size: {len(testing_set)}")

    # test_output_pt_dir = ""
    # test_output = torch.load(test_output_pt_dir)

    diffusion_model = Diffusion(num_timesteps=1000,
                                ddim_num_steps=100,
                                n_head=cfg.n_head,
                                dim_model=cfg.d_model,
                                feature_dim=cfg.feature_dim,
                                seq_dim=cfg.num_class + 4,
                                num_layers=cfg.n_layers,
                                device=device,
                                max_elem=cfg.max_elem,)
    # model_weights = torch.load(opt.check_path ,map_location=device)
    # diffusion_model.model.load_state_dict(model_weights)
    diffusion_model.model.eval()

    if cfg.task == 'uncond':
        test_output = sample_uncond(diffusion_model, testing_dl, cfg)
    elif cfg.task == 'refine':
        test_output, test_output_noise, test_output_gt = sample_refine(diffusion_model, testing_dl, cfg)
    else:
        test_output = sample_cond(diffusion_model, testing_dl, cfg, cond=cfg.task)

    img_names = torch.load(cfg.imgname_order_dir)
    img_names = img_names[:test_output.shape[0]]
    # occ_matrix = torch.load("")
    # rea_matrix = torch.load("")

    metrics = metric(img_names, test_output, cfg)

    # visualize
    cfg.save_imgs_dir = os.path.join(cfg.save_imgs_dir, f'/{opt.dataset}_{opt.anno}_test')
    visualize_images(img_names, test_output, cfg)

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
        '--dataset',
        type=str,
        default='pku',
        help='choose dataset to test (pku, cgl)')
    parser.add_argument(
        '--anno',
        type=str,
        default='unanno',
        help='choose dataset to test (anno, unanno)')
    parser.add_argument(
        '--task',
        type=str,
        default='uncond',
        help='choose task to test (uncond, c, cwh, complete, refinement)'
    )
    parser.add_argument(
        '--check_path',
        type=str,
        default='',
        help='choose checkpoint'
    )
    opt = parser.parse_args()
    main(opt)


# for idx in range(model_output_gt.shape[0]):
#     image_path = os.path.join(test_inp_dir, names[idx])
#     img = Image.open(image_path).convert("RGB")
#     res = model_output_gt[idx]
#     cls = res[:,:1]
#     box = res[:,1:]
#     draw_single(box, cls, img, idx, save_dir_1, width, height, num_class)
#
#     print(idx)