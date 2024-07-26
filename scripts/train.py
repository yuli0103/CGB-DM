import os
import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
from transformers import set_seed

import torch
from torch.utils.data import DataLoader
from data_process.dataloader import train_dataset, test_uncond_dataset, test_cond_dataset
from scripts.train_util import TrainLoop
from cgbdm.diffusion import Diffusion
from utils import logger
from utils.util import get_parameter_number, load_config

CUDA_LAUNCH_BLOCKING=1
# sys.path.append('/home/ly24/code/ditl')

def main(opt):
    seed = 1
    set_seed(seed)

    device = torch.device(f"cuda:{opt.gpuid}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # config_module = importlib.import_module(f'configs/config_{opt.dataset}')
    # cfg = config_module.config

    cfg = load_config(f'configs/{opt.dataset}.yaml')
    cfg.task = opt.task

    training_set = train_dataset(cfg)
    training_dl = DataLoader(training_set, num_workers=cfg.num_workers, batch_size=cfg.train_batch_size, shuffle=True)
    if cfg.task == 'uncond':
        cfg.imgname_order_dir = os.path.join(cfg.imgname_order_dir, f'seed_{seed}_{opt.dataset}_unanno_test.pt')
        evaling_set = test_uncond_dataset(cfg)
    else:
        cfg.imgname_order_dir = os.path.join(cfg.imgname_order_dir, f'seed_{seed}_{opt.dataset}_anno_test.pt')
        evaling_set = test_cond_dataset(cfg)
    evaling_dl = DataLoader(evaling_set, num_workers=cfg.num_workers, batch_size=cfg.test_batch_size, shuffle=False)

    logger.info(f"Training set size: {len(training_set)}, Evaling set size:{len(evaling_set)}")

    diffusion_model = Diffusion(num_timesteps=1000,
                                ddim_num_steps=100,
                                n_head=cfg.n_head,
                                dim_model=cfg.d_model,
                                feature_dim=cfg.feature_dim,
                                seq_dim=cfg.num_class + 4,
                                num_layers=cfg.n_layers,
                                device=device,
                                max_elem=cfg.max_elem,)
    total_num, trainable_num = get_parameter_number(diffusion_model.model)
    logger.info(f"trainable_num/total_num: %.2fM/%.2fM" % (trainable_num / 1e6, total_num / 1e6))

    # weights = torch.load('')
    # model_ddpm.model.load_state_dict(weights)
    TrainLoop(
        cfg,
        diffusion_model=diffusion_model,
        training_dl=training_dl,
        testing_dl=None,
        evaling_dl=evaling_dl,
        device=device,
    ).run_loop()

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
        help='choose dataset to train')
    parser.add_argument(
        '--task',
        type=str,
        default='uncond',
        help='choose task to train(uncond,c,cwh,complete)'
    )
    opt = parser.parse_args()
    main(opt)
