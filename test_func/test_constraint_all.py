import torch
import os
from utils.metric import metric
from utils import logger
from module.model_diffusion import Diffusion
from transformers import set_seed
from torch.utils.data import DataLoader
from utils.dataloader import canvas_train
from test_constraint import sample_r, sample_c

device = torch.device(f"cuda:{7}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
max_elem = 16

def cal_cls_error(model_output, gt_cls):
    pre_cls = model_output[:, :, :1]
    pre_cls = pre_cls.squeeze(-1)
    diff = torch.sum(pre_cls != gt_cls)
    total = torch.numel(gt_cls)
    error_rate = diff.float() / total
    print(f"Total elements: {total}")
    print(f"Total different elements: {diff}")
    print(f"Error rate: {error_rate:.4f}")
    print("---------------------------------")

def sample_constraint(gt_cls, epoch, test_data, num_class, model_ddpm, test_img_path, test_sal_path, test_sal_sub_path, img_path, writer):
    model_output_c = sample_c(model_ddpm, test_data, num_class, cond='c')
    model_output_cwh = sample_c(model_ddpm, test_data, num_class, cond='cwh')
    model_output_com = sample_c(model_ddpm, test_data, num_class, cond='com')
    model_output_r = sample_r(model_ddpm, test_data, num_class)

    dir = '/home/kl23/code/ditl/ptfile/pku/output/04_26_01'
    dir = dir +  'Epoch_{}/'.format(epoch)
    os.makedirs(dir, exist_ok=True)

    dir_c = dir + 'model_output_test_c.pt'
    torch.save(model_output_c, dir_c)
    dir_cwh = dir + 'model_output_test_cwh.pt'
    torch.save(model_output_cwh, dir_cwh)
    dir_com = dir + 'model_output_test_com.pt'
    torch.save(model_output_com, dir_com)
    dir_r = dir + 'model_output_test_r.pt'
    torch.save(model_output_r, dir_r)

    names = torch.load('ptfile/test_order_pku_split_test.pt')
    names = names[:model_output_r.shape[0]]

    cal_cls_error(model_output_c, gt_cls)
    metric_c = metric(names, model_output_c, test_img_path, test_sal_path, test_sal_sub_path,w=513, h=750)
    logger.log(f"Sample c {epoch} epoch done!")

    cal_cls_error(model_output_cwh, gt_cls)
    metric_cwh = metric(names, model_output_cwh, test_img_path, test_sal_path, test_sal_sub_path,w=513, h=750)
    logger.log(f"Sample cwh {epoch} epoch done!")

    cal_cls_error(model_output_com, gt_cls)
    metric_com = metric(names, model_output_com, test_img_path, test_sal_path, test_sal_sub_path,w=513, h=750)
    logger.log(f"Sample com {epoch} epoch done!")

    cal_cls_error(model_output_r, gt_cls)
    metric_r = metric(names, model_output_r, test_img_path, test_sal_path, test_sal_sub_path,w=513, h=750)
    logger.log(f"Sample r {epoch} epoch done!")

    writer.add_scalars('Val', {'c': metric_c['val'], 'cwh': metric_cwh['val'], 'com': metric_com['val'], 'r': metric_r['val']}, epoch)
    writer.add_scalars('Ali', {'c': metric_c['ali'], 'cwh': metric_cwh['ali'], 'com': metric_com['ali'], 'r': metric_r['ali']}, epoch)
    writer.add_scalars('Ove', {'c': metric_c['ove'], 'cwh': metric_cwh['ove'], 'com': metric_com['ove'], 'r': metric_r['ove']}, epoch)
    writer.add_scalars('Und_l', {'c': metric_c['undl'], 'cwh': metric_cwh['undl'], 'com': metric_com['undl'], 'r': metric_r['undl']},epoch)
    writer.add_scalars('Und_s', {'c': metric_c['unds'], 'cwh': metric_cwh['unds'], 'com': metric_com['unds'], 'r': metric_r['unds']},epoch)
    writer.add_scalars('Uti', {'c': metric_c['uti'], 'cwh': metric_cwh['uti'], 'com': metric_com['uti'], 'r': metric_r['uti']},epoch)
    writer.add_scalars('Occ', {'c': metric_c['occ'], 'cwh': metric_cwh['occ'], 'com': metric_com['occ'], 'r': metric_r['occ']},epoch)


def main():
    set_seed(42)
    width = 513
    height = 750
    test_batch_size = 64

    test_bg_dir = "/mnt/data/kl23/pku/split/test/inpaint"
    test_sal_dir = "/mnt/data/kl23/pku/split/test/sal"
    test_sal_sub_dir = ""
    test_gt_dir = "/mnt/data/kl23/pku/split/csv/test.csv"
    test_box_path = "/mnt/data/kl23/pku/split/csv/test_sal.csv"
    test_order_dir = "/home/kl23/code/ditl/ptfile/test_order_pku_split_test.pt"
    checkpoint_path = "/mnt/data/kl23/checkpoint/pku/04_24_04"

    weight_files = os.listdir(checkpoint_path)
    weight_files.sort(key=lambda x: int(x.split('_')[0][5:]))

    dataset_cls = "pku"
    if dataset_cls == 'pku':
        num_class = 4
    else:
        num_class = 5
    testing_set = canvas_train(test_bg_dir, test_sal_dir, test_sal_sub_dir, test_gt_dir, test_box_path,
                               num_class=num_class, max_elem=max_elem, dataset=dataset_cls,
                               train_if=False)
    testing_dl = DataLoader(testing_set, num_workers=16, batch_size=test_batch_size, shuffle=False)
    print(f"Testing set size: {len(testing_set)}")

    # writer = SummaryWriter()

    model_ddpm = Diffusion(num_timesteps=1000, n_head=8, dim_transformer=512,
                           feature_dim=1024, seq_dim=num_class+4, num_layers=4,
                           device=device, ddim_num_steps=100, max_elem=max_elem,
                           beta_schedule='cosine')
    for weight_file in weight_files:
        epoch = int(weight_file.split('_')[0][5:])
        if epoch==350:
            # continue
            checkpoint = os.path.join(checkpoint_path, weight_file)
            weights = torch.load(checkpoint ,map_location=device)
            model_ddpm.model.load_state_dict(weights)
            model_ddpm.model.eval()

            sample_constraint(epoch, testing_dl, num_class, model_ddpm, test_bg_dir, test_sal_dir, testing_set.inp, writer=None)
            break

    # writer.close()

    # names = torch.load(test_order_dir)
    # names = names[:model_output.shape[0]]
    # metric(names, model_output, test_bg_dir, test_sal_dir, w=width, h=height)

if __name__ == "__main__":
    main()