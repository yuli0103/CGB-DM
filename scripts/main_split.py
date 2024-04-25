#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from utils.dataloader import canvas_train, canvas_test
from transformers import set_seed
from train_util import TrainLoop
from module.model_diffusion import Diffusion

seed = 42
set_seed(seed)
device = torch.device(f"cuda:{5}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
CUDA_LAUNCH_BLOCKING=1
# devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

def main():
    max_elem = 16
    initial_lr = 1e-4
    total_epochs = 500
    gradient_clipping = 1.0

    d_model = 512
    n_head = 8
    n_layers = 4
    feature_dim = d_model * 2
    beta_schedule = 'cosine'
    date = "04_26_02"

    dataset_cls = 'pku'
    if dataset_cls=='pku':
        num_class = 4
    else:
        num_class = 5

    train_batch_size = 32
    train_inp_dir = "/mnt/data/kl23/pku/split/train/inpaint"
    train_sal_dir = "/mnt/data/kl23/pku/split/train/saliency"
    train_sal_sub_dit = "/mnt/data/kl23/pku/split/train/saliency_sub"
    train_csv_path = "/mnt/data/kl23/pku/split/csv/train.csv"
    train_box_path = "/mnt/data/kl23/pku/split/csv/train_sal.csv"
    training_set = canvas_train(train_inp_dir, train_sal_dir, train_sal_sub_dit, train_csv_path, train_box_path,
                                max_elem, num_class, dataset=dataset_cls)
    training_dl = DataLoader(training_set, num_workers=16, batch_size=train_batch_size, shuffle=True)

    val_batch_size = 64
    val_inp_dir = "/mnt/data/kl23/pku/nosplit/test/image_canvas"
    val_sal_dir = "/mnt/data/kl23/pku/nosplit/test/saliency"
    val_sal_sub_dir = "/mnt/data/kl23/pku/nosplit/test/saliency_sub"
    val_box_path = "/mnt/data/kl23/pku/nosplit/csv/test_sal.csv"
    valing_set = canvas_test(val_inp_dir, val_sal_dir, val_sal_sub_dir, val_box_path, max_elem, dataset=dataset_cls)
    valing_dl = DataLoader(valing_set, num_workers=16, batch_size=val_batch_size, shuffle=False)

    test_batch_size = 64
    test_bg_dir = "/mnt/data/kl23/pku/split/test/inpaint"
    test_sal_dir = "/mnt/data/kl23/pku/split/test/saliency"
    test_sal_sub_dir = "/mnt/data/kl23/pku/split/test/saliency_sub"
    test_box_path = "/mnt/data/kl23/pku/split/csv/test_sal.csv"
    testing_set = canvas_test(test_bg_dir, test_sal_dir, test_sal_sub_dir, test_box_path,
                               max_elem, dataset=dataset_cls)
    testing_dl = DataLoader(testing_set, num_workers=16, batch_size=test_batch_size, shuffle=False)

    image_path = {}
    image_path['test_img'] = test_bg_dir
    image_path['test_sal'] = test_sal_dir
    image_path['test_sal_sub'] = test_sal_sub_dir
    image_path['val_img'] = val_inp_dir
    image_path['val_sal'] = val_sal_dir
    image_path['val_sal_sub'] = val_sal_sub_dir
    print(f"Training set size: {len(training_set)}, Valing set size:{len(valing_set)}, Testing set size:{len(testing_set)}")

    model_ddpm = Diffusion(num_timesteps=1000, n_head=n_head, dim_transformer=d_model,
                           feature_dim=feature_dim, seq_dim=num_class + 4, num_layers=n_layers,
                           device=device, ddim_num_steps=100, max_elem=max_elem,
                           beta_schedule=beta_schedule)
    # total_num, trainable_num = get_parameter_number(model_ddpm.model)
    # print(f"trainable_num/total_num: %.2fM/%.2fM" % (trainable_num / 1e6, total_num / 1e6))
    # sys.exit()

    # 训练集 & 验证集
    # weights = torch.load('/home/ly24/code/py_model/checkpoint/03_21_01/Epoch500_model_weights.pth')
    # model_ddpm.model.load_state_dict(weights)
    TrainLoop(
        model_ddpm=model_ddpm,
        training_dl=training_dl,
        testing_dl=testing_dl,
        valing_dl=valing_dl,
        train_batch_size=train_batch_size,
        initial_lr=initial_lr,
        gradient_clipping=gradient_clipping,
        total_epochs=total_epochs,
        max_elem=max_elem,
        num_class=num_class,
        device=device,
        paths=image_path,
        date=date,
    ).run_loop()


if __name__ == "__main__":
    main()
