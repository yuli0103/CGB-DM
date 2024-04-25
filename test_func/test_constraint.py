from PIL import Image
import torch
from utils import logger
import os
from utils.dataloader import canvas_train
from torch.utils.data import DataLoader
from module.model_diffusion import Diffusion
from transformers import set_seed
from utils.metric import metric

device = torch.device(f"cuda:{6}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
max_elem = 16

def finalize(layout, num_class):
    bbox = layout[:, :, num_class:]
    bbox = torch.clamp(bbox, min=-1, max=1) / 2 + 0.5
    label = torch.argmax(layout[:, :, :num_class], dim=2)
    mask = (label != 0).clone().detach()
    label = label.unsqueeze(-1)
    return bbox, label, mask

def sample_c(model_ddpm, testing_dl, num_class=4, cond='c', img_size=(513,750)):
    model_output = []
    width, height = img_size
    device = model_ddpm.device
    cnt = 0
    for idx, (image, layout, detect_box) in enumerate(testing_dl):
        image, layout, detect_box= image.to(device), layout.to(device), detect_box.to(device)
        layout[:, :, num_class:] = 2 * (layout[:, :, num_class:] - 0.5)
        detect_box = 2 * (detect_box - 0.5)

        box, cls, mask = model_ddpm.conditional_reverse_ddim(layout, image, detect_box, cond)
        samples = torch.cat([cls, box], dim=2)
        model_output.append(samples.cpu())

        cnt = cnt + image.shape[0]
        logger.log(f"created {cnt} samples")
    model_output = torch.concat(model_output, dim=0)
    return model_output

def sample_r(model_ddpm, testing_dl, num_class=4, img_size=(513,750)):
    width, height = img_size
    model_output = []
    cnt = 0

    for idx, (image, layout, detect_box) in enumerate(testing_dl):
        image, layout, detect_box= image.to(device), layout.to(device), detect_box.to(device)
        layout[:, :, num_class:] = 2 * (layout[:, :, num_class:] - 0.5)
        cls_onehot = layout[:,:,:num_class]
        box_gt, cls_gt, mask_gt = finalize(layout, num_class)
        box_noise = torch.clamp(box_gt + 0.01 * torch.randn_like(box_gt), min=0, max=1)

        box_noise = 2 * (box_noise - 0.5)
        noise_layout = torch.cat((cls_onehot, box_noise), dim=2).to(device)
        detect_box = 2 * (detect_box - 0.5)
        box, cls, mask = model_ddpm.refinement_reverse_ddim(noise_layout, image, detect_box)

        samples = torch.cat([cls, box], dim=2)
        model_output.append(samples.cpu())

        cnt = cnt + image.shape[0]
        logger.log(f"created {cnt} samples")

    model_output = torch.concat(model_output, dim=0)
    return model_output

def main():
    set_seed(42)
    width = 513
    height = 750
    test_batch_size = 64

    test_bg_dir = "/mnt/data/kl23/pku/split/test/inpaint"
    test_sal_dir = "/mnt/data/kl23/pku/split/test/sal"
    test_gt_dir = "/mnt/data/kl23/pku/split/csv/test.csv"
    test_box_path = "/mnt/data/kl23/pku/split/csv/test_sal.csv"

    test_order_dir = "/home/kl23/code/ditl/ptfile/test_order_pku_split_test.pt"
    model_output_dir = "/home/kl23/code/ditl/ptfile/pku/output/04_21_03/Epoch_480/model_output_test.pt"
    checkpoint_path = "/mnt/data/kl23/checkpoint/pku/04_21_04/Epoch383_model_weights.pth"

    dataset_cls = "pku"
    if dataset_cls == 'pku':
        num_class = 4
    else:
        num_class = 5
    testing_set = canvas_train(test_bg_dir, test_sal_dir, test_gt_dir, test_box_path,
                               num_class=num_class, max_elem=max_elem, dataset=dataset_cls,
                               train_if=False)
    testing_dl = DataLoader(testing_set, num_workers=16, batch_size=test_batch_size, shuffle=False)
    print(f"Testing set size: {len(testing_set)}")

    model_ddpm = Diffusion(num_timesteps=1000, n_head=8, dim_transformer=512,
                           feature_dim=1024, seq_dim=num_class+4, num_layers=4,
                           device=device, ddim_num_steps=100    , max_elem=max_elem,
                           beta_schedule='linear')

    weights = torch.load(checkpoint_path ,map_location=device)
    model_ddpm.model.load_state_dict(weights)
    model_ddpm.model.eval()

    model_output = sample_c(model_ddpm, testing_dl, num_class=num_class, cond="cwh", img_size=(width,height))

    model_output = sample_r(model_ddpm, testing_dl, num_class=num_class, img_size=(width,height))

    # model_output = torch.load(model_output_dir)

    names = torch.load(test_order_dir)
    names = names[:model_output.shape[0]]
    metric(names, model_output, test_bg_dir, test_sal_dir, w=width, h=height)


if __name__ == "__main__":
    main()