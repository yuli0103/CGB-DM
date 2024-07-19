import sys
import os

import numpy as np
import torch
import torch.nn as nn
from cgbdm.diffusion_utils import *
from cgbdm.layout_model import LayoutModel
from utils.visualize import draw_image
from utils.util import finalize

class Diffusion(nn.Module):
    def __init__(self,
                 num_timesteps=1000,
                 ddim_num_steps=100,
                 n_head=8,
                 dim_model=512,
                 feature_dim=2048,
                 seq_dim=8,
                 num_layers=4,
                 device='cuda',
                 max_elem=16):
        super().__init__()
        self.device = device
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(schedule='cosine', num_timesteps=self.num_timesteps)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coeff_2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.logvar = betas.log()

        self.seq_dim = seq_dim
        self.seq_len = max_elem
        self.num_class = seq_dim - 4

        self.model = LayoutModel(num_layers=num_layers, dim_seq=seq_dim,
                                 dim_model=dim_model, n_head=n_head,
                                 dim_feedforward=feature_dim, diffusion_steps=num_timesteps,
                                 max_elem=max_elem, device=device).to(self.device)

        self.ddim_num_steps = ddim_num_steps
        self.make_ddim_schedule(ddim_num_steps)
        self.make_ddim_refine_schedule(ddim_num_steps)

    def make_ddim_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps)

        betas_ddim = make_beta_schedule(schedule='linear', num_timesteps=self.num_timesteps)
        betas_ddim = betas_ddim.float().to(self.device)
        alphas_ddim = 1.0 - betas_ddim
        self.alphas_cumprod_ddim = alphas_ddim.cumprod(dim=0)
        assert self.alphas_cumprod_ddim.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('sqrt_alphas_cumprod', to_torch(torch.sqrt(self.alphas_cumprod_ddim)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(torch.sqrt(1. - self.alphas_cumprod_ddim)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(torch.log(1. - self.alphas_cumprod_ddim)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod_ddim)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod_ddim - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod_ddim,
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', torch.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def make_ddim_refine_schedule(self, ddim_num_steps, ddim_discretize="refine", ddim_eta=0.):
        self.ddim_refine_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps)

        # ddim sampling parameters
        ddim_refine_sigmas, ddim_refine_alphas, ddim_refine_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod_ddim,
                                                                                   ddim_timesteps=self.ddim_refine_timesteps,
                                                                                   eta=ddim_eta)
        self.register_buffer('ddim_refine_sigmas', ddim_refine_sigmas)
        self.register_buffer('ddim_refine_alphas', ddim_refine_alphas)
        self.register_buffer('ddim_refine_alphas_prev', ddim_refine_alphas_prev)
        self.register_buffer('ddim_refine_sqrt_one_minus_alphas', torch.sqrt(1. - ddim_refine_alphas))

    def load_diffusion_net(self, net_state_dict):
        new_states = dict()
        for k in net_state_dict.keys():
            if 'layer_out' not in k and 'layer_in' not in k:
                new_states[k] = net_state_dict[k]
        self.model.load_state_dict(net_state_dict, strict=True)

    def sample_t(self, size=(1,), t_max=None):
       """Samples batches of time steps to use."""
       if t_max is None:
           t_max = int(self.num_timesteps) - 1
       t = torch.randint(low=0, high=t_max, size=size, device=self.device)
       return t.to(self.device)

    def sample_t_temp(self, size=(1,), t_max=None):
        """Samples batches of time steps to use."""
        if t_max is None:
            t_max = int(self.num_timesteps)
        size = (size,) if isinstance(size, int) else size
        t = torch.zeros(size + (t_max,), dtype=torch.long, device=self.device)
        for i in range(t_max):
            t[..., i] = i
        return t

    def forward_t(self, l_0_batch, image, sal_box, t, cond='uncond', reparam=False):
        batch_size = l_0_batch.shape[0]
        e = torch.randn_like(l_0_batch).to(l_0_batch.device)

        fix_mask = torch.zeros([batch_size, self.seq_len, self.seq_dim]).to(torch.bool).to(self.device)
        if cond == 'c':
            fix_ind = [x for x in range(self.num_class)]
            fix_mask[:, :, fix_ind] = True
            e[fix_mask] = 0
        elif cond == 'cwh':
            fix_ind = [x for x in range(self.num_class)] + [self.num_class + 2, self.num_class + 3]
            fix_mask[:, :, fix_ind] = True
            e[fix_mask] = 0
        elif cond == 'complete':
            real_label = torch.argmax(l_0_batch[:, :, :self.num_class], dim=2)
            real_mask = (real_label != 0).clone().detach()
            temp_mask = rand_fix(batch_size, real_mask, ratio=0.2, n_elements=self.seq_len)
            fix_mask[:, :, :self.seq_dim] = temp_mask.unsqueeze(-1).expand(-1, -1, self.seq_dim)
            e[fix_mask] = 0

        l_t_noise = q_sample(l_0_batch, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, fix_mask, t, noise=e, cond=cond)

        eps_theta = self.model(l_t_noise, image, sal_box, timestep=t)

        if reparam:
            sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, l_t_noise)
            sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
            l_0_generate_reparam = 1 / sqrt_alpha_bar_t * (l_t_noise - eps_theta * sqrt_one_minus_alpha_bar_t).to(self.device)

            return eps_theta, e, l_0_generate_reparam
        else:
            return eps_theta, e

    def reverse(self, batch_size, only_last_sample=True, stochastic=True):

        self.model.eval()
        layout_t_0 = p_sample_loop(self.model, batch_size,
                                  self.num_timesteps, self.alphas,
                                  self.one_minus_alphas_bar_sqrt,
                                  only_last_sample=only_last_sample, stochastic=stochastic)

        bbox, label, mask = finalize(layout_t_0, self.num_class)

        return bbox, label, mask

    def reverse_ddim(self, image, sal_box, cfg, save_inter_dir='', img=None, save_inter=False):
        self.model.eval()
        layout_t_0, intermediates = ddim_sample_loop(self.model, image, sal_box, self.ddim_timesteps, self.ddim_alphas,
                                                     self.ddim_alphas_prev, self.ddim_sigmas,
                                                     seq_len=cfg.max_elem, seq_dim=self.seq_dim)
        bbox, label, mask = finalize(layout_t_0, self.num_class)

        if save_inter:
            for i, layout_t in enumerate(intermediates['y_inter']):
                bbox_inter, label_inter, mask_inter = finalize(layout_t, self.num_class)
                draw_image(bbox_inter, label_inter, img, i, width=cfg.width, height=cfg.height,
                           numclass=self.num_class, save_dir=save_inter_dir)
        return bbox, label, mask

    def conditional_reverse_ddim(self, real_layout, image, sal_box, cfg, save_inter_dir='', img=None,
                                 cond='c', ratio=0.1, stochastic=True, save_inter=False):

        self.model.eval()
        layout_t_0, intermediates = \
            ddim_cond_sample_loop(self.model, real_layout, image, sal_box, self.ddim_timesteps, self.ddim_alphas,
                                  self.ddim_alphas_prev, self.ddim_sigmas, stochastic=stochastic, cond=cond,
                                  ratio=ratio)

        bbox, label, mask = finalize(layout_t_0, self.num_class)

        if save_inter:
            for i, layout_t in enumerate(intermediates['y_inter']):
                bbox_inter, label_inter, mask_inter = finalize(layout_t, self.num_class)
                draw_image(bbox_inter, label_inter, img, i, width=cfg.width, height=cfg.height,
                           numclass=self.num_class, save_dir=save_inter_dir)

        return bbox, label, mask

    def refinement_reverse_ddim(self, noisy_layout, image, sal_box):
        self.model.eval()

        layout_t_0, intermediates = \
            ddim_refine_sample_loop(self.model, noisy_layout, image, sal_box, self.ddim_refine_timesteps, self.ddim_refine_alphas,
                                    self.ddim_refine_alphas_prev, self.ddim_refine_sigmas)

        bbox, label, mask = finalize(layout_t_0, self.num_class)

        return bbox, label, mask

