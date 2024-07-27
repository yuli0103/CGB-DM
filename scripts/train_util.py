import os
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import logger
import torch.nn as nn
from utils.metric import metric
from scripts.test import sample_uncond, sample_cond


class TrainLoop:
    def __init__(
            self,
            cfg,
            diffusion_model,
            training_dl,
            testing_dl,
            evaling_dl,
            device,
    ):
        self.datetime=cfg.datetime
        self.diffusion_model = diffusion_model

        self.train_data = training_dl
        self.val_data = evaling_dl
        self.test_data = testing_dl
        self.cfg = cfg

        self.initial_lr = cfg.lr
        self.gradient_clipping = cfg.gradient_clipping
        self.epochs = cfg.epochs
        self.num_class = cfg.num_class

        self.master_params = list(self.diffusion_model.model.parameters())
        self.device = device

        self.opt = optim.Adam(self.master_params, lr=self.initial_lr, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs)
        self.writer = SummaryWriter()

    def requires_grad(self, model, flag=False):
        """
        Set requires_grad flag for all parameters in a model.
        """
        for p in model.parameters():
            p.requires_grad = flag

    def get_description(self, epoch, epochs, lr, loss):
        return (f'Epoch {epoch} / Epochs {epochs}, '
                f'LR: {lr:.2e}, '
                f'Loss: {loss:.4f}')

    def optimize_normal(self):
        if self.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(self.diffusion_model.model.parameters(), self.gradient_clipping)
        self.opt.step()

    def log_metrics(self, metric_res, epoch):
        metrics = {
            'Val': 'val',
            'Ove': 'ove',
            'Und_l': 'undl',
            'Und_s': 'unds',
            'Rea': 'rea',
            'Occ': 'occ'
        }
        for display_name, metric_key in metrics.items():
            self.writer.add_scalar(display_name, metric_res[metric_key], epoch)

    def test_uncond(self):
        test_output = sample_uncond(self.diffusion_model, self.val_data, self.cfg)
        img_names = torch.load(self.cfg.imgname_order_dir)

        # load matrix infomation
        # occ_matrix = torch.load("")
        # rea_matrix = torch.load("")

        metrics = metric(img_names, test_output, self.cfg)

        # store sample output
        # base_test_output_dir = Path('')
        # test_output_dir = base_test_output_dir / self.datetime
        # test_output_dir.mkdir(parents=True, exist_ok=True)
        # test_output_dir = test_output_dir + 'test_output.pt'
        # torch.save(test_output, test_output_dir)
        return metrics

    def test_constraint(self,):
        cond = self.cfg.task
        # occ_matrix = torch.load("")
        # rea_matrix = torch.load("")

        test_output = sample_cond(self.diffusion_model, self.val_data, self.cfg, cond=cond)
        img_names = torch.load(self.cfg.imgname_order_dir)
        metrics = metric(img_names, test_output, self.cfg)

        # store sample output
        # output_dir = Path('') / self.cfg.task / self.datetime
        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_path = output_dir / 'test_output.pt'
        # torch.save(test_output, output_path)
        return metrics


    def run_loop(self):
        logger.info(f"Training for {self.epochs} epochs...")
        base_check_dir = Path(self.cfg.base_check_dir)
        check_dir = base_check_dir / self.datetime
        check_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):
            epoch += 1
            self.run_train_step(self.train_data, epoch)
            logger.info("train finish!")
            # Modify log_test_epochs, observe the validation set results on tensorboard, and select the optimal weight
            if epoch>=400 and epoch % self.cfg.log_test_epochs == 0:
                if self.cfg.task == 'uncond':
                    metrics = self.test_uncond()
                else:
                    metrics = self.test_constraint()
                self.log_metrics(metrics, epoch)
                logger.log(f"Sample {self.cfg.task} {epoch} epoch done!")

                file_name = f'Epoch{epoch}_cgbdm_weights.pth'
                check_epoch_dir = os.path.join(check_dir, file_name)
                torch.save(self.diffusion_model.model.state_dict(), check_epoch_dir)

            self.scheduler.step()
        logger.info("Done!")
        # torch.save(self.diffusion_model.model.state_dict(), check_dir)
        self.writer.close()

    def run_train_step(self, data, epoch):
        steps = 0
        total_loss = 0
        mse_loss = nn.MSELoss()
        pbar = tqdm(data, desc=f'Epoch {epoch}')

        for idx, (image, layout, sal_box) in enumerate(pbar):
            self.opt.zero_grad()
            image, layout, sal_box = image.to(self.device), layout.to(self.device), sal_box.to(self.device)
            t = self.diffusion_model.sample_t([layout.shape[0]], t_max=self.diffusion_model.num_timesteps - 1)

            eps_theta, e= self.diffusion_model.forward_t(layout, image, sal_box, t=t, cond=self.cfg.task)
            loss = mse_loss(e, eps_theta)
            total_loss += loss
            steps += 1

            description = self.get_description(epoch, self.epochs, self.opt.param_groups[0]["lr"], total_loss / steps)
            pbar.set_description(description)
            loss.backward()
            self.optimize_normal()

        logger.log(description)
        self.writer.add_scalar('Loss/train', total_loss / steps, epoch)






