import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import logger
import torch.nn as nn
from utils.metric import metric
from test_func.test import sample
from test_func.test_constraint import sample_c, sample_r


class TrainLoop:
    def __init__(
            self,
            *,
            model_ddpm,
            training_dl,
            testing_dl,
            valing_dl,
            train_batch_size,
            initial_lr,
            gradient_clipping,
            total_epochs,
            max_elem,
            num_class,
            device,
            paths,
            date,
    ):
        self.date=date
        self.model_ddpm = model_ddpm

        self.data = training_dl
        self.val_data = valing_dl
        self.test_data = testing_dl
        self.data_length = len(training_dl)
        self.batch_size = train_batch_size
        self.test_img_path = paths['test_img']
        self.test_sal_path = paths['test_sal']
        self.test_sal_sub_path = paths['test_sal_sub']
        self.val_img_path = paths['val_img']
        self.val_sal_path = paths['val_sal']
        self.val_sal_sub_path = paths['val_sal_sub']

        self.initial_lr = initial_lr
        self.gradient_clipping = gradient_clipping
        self.total_epochs = total_epochs
        self.max_elem = max_elem
        self.num_class = num_class

        self.master_params = list(self.model_ddpm.model.parameters())
        self.device = device
        # self.ema = deepcopy(self.model_ddpm.model).to(device)

        self.opt = optim.Adam(self.master_params, lr=self.initial_lr, weight_decay=0.0, betas=(0.9, 0.999),
                                                                 amsgrad=False, eps=1e-08)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.total_epochs)
        self.writer = SummaryWriter()
        # self.checkpoint_path = checkpoint_path  # DEBUG **

    def requires_grad(self, model, flag=False):
        """
        Set requires_grad flag for all parameters in a model.
        """
        for p in model.parameters():
            p.requires_grad = flag

    def test_val(self,epoch=-1):
        model_output = sample(self.model_ddpm, self.val_data)
        dir = '/home/kl23/code/ditl/ptfile/pku/output/'
        dir = dir +'{}/'.format(self.date) +'Epoch_{}/'.format(epoch)
        os.makedirs(dir, exist_ok=True)
        dir = dir + 'model_output_val.pt'
        torch.save(model_output, dir)

        names = torch.load('/home/kl23/code/ditl/ptfile/test_order_pku.pt')
        names = names[:model_output.shape[0]]
        logger.log(f"Sample {epoch} epoch done!")

        metric_r = metric(names, model_output,self.val_img_path, self.val_sal_path,self.val_sal_sub_path,w=513, h=750)
        self.writer.add_scalar('Val_v', metric_r['val'], epoch)
        self.writer.add_scalar('Ali_v', metric_r['ali'], epoch)
        self.writer.add_scalar('Ove_v', metric_r['ove'], epoch)
        self.writer.add_scalar('Und_l_v', metric_r['undl'], epoch)
        self.writer.add_scalar('Und_s_v', metric_r['unds'], epoch)
        self.writer.add_scalar('Uti_v', metric_r['uti'], epoch)
        self.writer.add_scalar('Occ_v', metric_r['occ'], epoch)
        # self.writer.add_scalar('Rea', rea, epoch)

    def test(self,epoch=-1):
        model_output = sample(self.model_ddpm, self.test_data)

        dir = '/home/kl23/code/ditl/ptfile/pku/output/'
        dir = dir +'{}/'.format(self.date) +'Epoch_{}/'.format(epoch)
        os.makedirs(dir, exist_ok=True)
        dir = dir + 'model_output_test.pt'
        torch.save(model_output, dir)

        names = torch.load('/home/kl23/code/ditl/ptfile/test_order_pku_split_test.pt')
        names = names[:model_output.shape[0]]
        logger.log(f"Sample {epoch} epoch done!")

        metric_r = metric(names, model_output,self.test_img_path, self.test_sal_path,self.test_sal_sub_path,w=513, h=750)
        self.writer.add_scalar('Val_t', metric_r['val'], epoch)
        self.writer.add_scalar('Ali_t', metric_r['ali'], epoch)
        self.writer.add_scalar('Ove_t', metric_r['ove'], epoch)
        self.writer.add_scalar('Und_l_t', metric_r['undl'], epoch)
        self.writer.add_scalar('Und_s_t', metric_r['unds'], epoch)
        self.writer.add_scalar('Uti_t', metric_r['uti'], epoch)
        self.writer.add_scalar('Occ_t', metric_r['occ'], epoch)
        # self.writer.add_scalar('Rea', rea, epoch)

    def test_constraint(self,epoch=-1):

        model_output_c = sample_c(self.model_ddpm, self.test_data,self.num_class, cond='c')
        model_output_cwh = sample_c(self.model_ddpm, self.test_data,self.num_class, cond='cwh')
        model_output_com = sample_c(self.model_ddpm, self.test_data,self.num_class, cond='com')
        model_output_r = sample_r(self.model_ddpm, self.test_data,self.num_class)

        names = torch.load('ptfile/test_order_pku_split_test.pt')
        names = names[:model_output_c.shape[0]]

        dir = '/home/kl23/code/ditl/ptfile/pku/output/'
        dir = dir + '{}/'.format(self.date) + 'Epoch_{}/'.format(epoch)
        os.makedirs(dir, exist_ok=True)

        dir_c = dir + 'model_output_test_c.pt'
        torch.save(model_output_c, dir_c)
        dir_cwh = dir + 'model_output_test_cwh.pt'
        torch.save(model_output_cwh, dir_cwh)
        dir_com = dir + 'model_output_test_com.pt'
        torch.save(model_output_com, dir_com)
        dir_r = dir + 'model_output_test_r.pt'
        torch.save(model_output_r, dir_r)

        # val,ali,ove,und_l,und_s,uti,occ = metric(names, model_output_c, self.test_img_path, self.test_sal_path, w=513, h=750)
        metric_c = metric(names, model_output_c,self.test_img_path,self.test_sal_path,self.test_sal_sub_path,w=513, h=750)
        logger.log(f"Sample c {epoch} epoch done!")
        metric_cwh = metric(names, model_output_cwh, self.test_img_path, self.test_sal_path, self.test_sal_sub_path, w=513,h=750)
        logger.log(f"Sample cwh {epoch} epoch done!")
        metric_com = metric(names, model_output_com, self.test_img_path, self.test_sal_path, self.test_sal_sub_path, w=513,h=750)
        logger.log(f"Sample complete {epoch} epoch done!")
        metric_r = metric(names, model_output_r, self.test_img_path, self.test_sal_path, self.test_sal_sub_path, w=513, h=750)
        logger.log(f"Sample refine {epoch} epoch done!")

        self.writer.add_scalars('Val', {'c': metric_c['val'], 'cwh': metric_cwh['val'], 'com': metric_com['val'], 'r': metric_r['val']}, epoch)
        self.writer.add_scalars('Ali', {'c': metric_c['ali'], 'cwh': metric_cwh['ali'], 'com': metric_com['ali'], 'r': metric_r['ali']}, epoch)
        self.writer.add_scalars('Ove', {'c': metric_c['ove'], 'cwh': metric_cwh['ove'], 'com': metric_com['ove'], 'r': metric_r['ove']}, epoch)
        self.writer.add_scalars('Und_l', {'c': metric_c['undl'], 'cwh': metric_cwh['undl'], 'com': metric_com['undl'], 'r': metric_r['undl']},epoch)
        self.writer.add_scalars('Und_s', {'c': metric_c['unds'], 'cwh': metric_cwh['unds'], 'com': metric_com['unds'], 'r': metric_r['unds']},epoch)
        self.writer.add_scalars('Uti', {'c': metric_c['uti'], 'cwh': metric_cwh['uti'], 'com': metric_com['uti'], 'r': metric_r['uti']},epoch)
        self.writer.add_scalars('Occ', {'c': metric_c['occ'], 'cwh': metric_cwh['occ'], 'com': metric_com['occ'], 'r': metric_r['occ']},epoch)

    def run_loop(self):
        logger.info(f"Training for {self.total_epochs} epochs...")

        check_dir = '/mnt/data/kl23/checkpoint/pku/'
        check_dir = check_dir + '{}/'.format(self.date)
        os.makedirs(check_dir, exist_ok=True)
        for epoch in range(self.total_epochs):
            epoch += 1
            self.run_train_step(self.data, epoch)
            logger.info("train finish!")
            if epoch>=330:
                if epoch>=350 and epoch<=500:
                    torch.save(self.model_ddpm.model.state_dict(), check_dir + 'Epoch{}_model_weights.pth'.format(epoch))
                if self.val_data is not None:
                    self.test_val(epoch=epoch)
                    logger.info("eval finish!")
                # self.test_constraint(epoch=epoch)
                self.test(epoch=epoch)
                logger.info("test finish!")
            self.scheduler.step()
        logger.info("Done!")
        self.writer.close()

    def run_train_step(self, data, epoch):
        steps = 0
        total_loss = 0
        mse_loss = nn.MSELoss()
        pbar = tqdm(data, desc=f'Epoch {epoch}')

        for idx, (image, layout, detect_box) in enumerate(pbar):
            self.opt.zero_grad()
            image, layout, detect_box = image.to(self.device), layout.to(self.device), detect_box.to(self.device)
            t = self.model_ddpm.sample_t([layout.shape[0]], t_max=self.model_ddpm.num_timesteps - 1)
            layout[:, :, self.num_class:] = 2 * (layout[:, :, self.num_class:] - 0.5)
            detect_box = 2 * (detect_box - 0.5)
            # imgs = []
            # for i in range(4):
            #     img = Image.open(self.img_path[idx * 4 + i]).convert("RGB").resize((513, 750))
            #     imgs.append(img)
            eps_theta, e= self.model_ddpm.forward_t(layout, image, detect_box, t=t, reparam=False, train=True)
            loss = mse_loss(e, eps_theta)
            total_loss += loss
            steps += 1
            pbar.set_description(f'Epoch {epoch} / Epochs {self.total_epochs}, '
                                 f'LR: {self.opt.param_groups[0]["lr"]:.2e}, '
                                 f'Loss: {total_loss / steps:.4f}')
            loss.backward()
            self.optimize_normal()
        logger.log(f'Epoch {epoch} / Epochs {self.total_epochs}, '
                   f'LR: {self.opt.param_groups[0]["lr"]:.2e}, '
                   f'Loss: {total_loss / steps:.4f} ')
        self.writer.add_scalar('Loss/train', total_loss / steps, epoch)

    def run_val_step(self, data, epoch):
        steps = 0
        total_loss = 0
        mse_loss = nn.MSELoss()
        with torch.no_grad():
            for idx, (image, layout, detect_box) in enumerate(data):
                image, layout, detect_box = image.to(self.device), layout.to(self.device), detect_box.to(self.device)
                t = self.model_ddpm.sample_t([layout.shape[0]], t_max=self.model_ddpm.num_timesteps - 1)

                layout[:, :, self.num_class:] = 2 * (layout[:, :, self.num_class:] - 0.5)
                detect_box = 2 * (detect_box - 0.5)
                eps_theta, e= self.model_ddpm.forward_t(layout, image, detect_box, t=t, real_mask=None, reparam=False, train=False)
                loss = mse_loss(e, eps_theta)
                total_loss += loss
                steps += 1

        logger.log(f'Epoch {epoch} / Epochs {self.total_epochs}, ' f'Loss: {total_loss / steps:.4f} ')
        self.writer.add_scalar('Loss/val', total_loss / steps, epoch)

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.model_ddpm.model.parameters(), self.gradient_clipping)
        self.opt.step()





