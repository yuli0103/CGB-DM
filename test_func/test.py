from PIL import Image
import torch
from utils import logger
import os
from utils.dataloader import canvas_test
from torch.utils.data import DataLoader
from utils.visual import draw_single
from module.model_diffusion import Diffusion
from transformers import set_seed
from utils.metric import metric

device = torch.device(f"cuda:{4}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
max_elem = 16

def sample(model_ddpm, testing_dl, img_size=(513,750)):
    width, height = img_size
    model_output = []
    device = model_ddpm.device
    cnt = 0
    for idx, (image, detect_box) in enumerate(testing_dl):
        image, detect_box= image.to(device), detect_box.to(device)
        detect_box = 2 * (detect_box - 0.5)
        bbox_generated, cls, mask = model_ddpm.reverse_ddim(image, detect_box,
                                                            stochastic=True,
                                                            save_inter=False,
                                                            max_len=max_elem)

        samples = torch.cat([cls, bbox_generated], dim=2)
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

    test_bg_dir = "/mnt/data/kl23/pku/nosplit/train/inpainted_images"
    test_sal_dir = "/mnt/data/kl23/pku/nosplit/train/saliency"
    test_sal_sub_dir = "/mnt/data/kl23/pku/nosplit/train/saliency_sub"
    test_box_path = "/mnt/data/kl23/pku/nosplit/csv/train_sal.csv"

    test_order_dir = "/home/kl23/code/ditl/ptfile/test_order_pku.pt"
    model_output_dir = "/home/kl23/code/ditl/ptfile/pku/output/04_24_01/Epoch_477/model_output_test.pt"
    checkpoint_path = '/mnt/data/kl23/checkpoint/pku/04_24_05/Epoch476_model_weights.pth'

    testing_set = canvas_test(test_bg_dir, test_sal_dir, test_sal_sub_dir,  test_box_path, max_elem=max_elem, dataset="pku")
    testing_dl = DataLoader(testing_set, num_workers=16, batch_size=test_batch_size, shuffle=False)
    print(f"Testing set size: {len(testing_set)}")

    model_ddpm = Diffusion(num_timesteps=1000, n_head=8, dim_transformer=512,
                           feature_dim=1024, seq_dim=8, num_layers=4,
                           device=device, ddim_num_steps=100, max_elem=max_elem,
                           beta_schedule='cosine')

    weights = torch.load(checkpoint_path ,map_location=device)
    model_ddpm.model.load_state_dict(weights)
    model_ddpm.model.eval()

    model_output = sample(model_ddpm, testing_dl, img_size=(width,height))
    # model_output = torch.load(model_output_dir)
    names = torch.load(test_order_dir)
    names = names[:model_output.shape[0]]
    metric(names, model_output, test_bg_dir, test_sal_dir, test_sal_sub_dir, w=width, h=height)


if __name__ == "__main__":
    main()