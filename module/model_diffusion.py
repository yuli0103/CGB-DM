import numpy as np
import torch
import torch.nn as nn
from module.diffusion_utils import *
from module.model import My_Model
import os
from utils.visual import draw_single

class Diffusion(nn.Module):
    def __init__(self, num_timesteps=1000, n_head=8, feature_dim=2048, dim_transformer=512, seq_dim=8, num_layers=4, device='cuda',
                 beta_schedule='cosine', ddim_num_steps=50,  max_elem=16):
        super().__init__()
        self.device = device
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=self.num_timesteps, start=2e-4, end=4e-2, ratio=1)
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
        self.num_class = seq_dim - 4

        self.model = My_Model(num_layers=num_layers, dim_seq=seq_dim,
                              dim_transformer=dim_transformer, n_head=n_head,
                              dim_feedforward=feature_dim, diffusion_step=num_timesteps,
                              max_elem=max_elem, device=device).to(self.device)

        self.ddim_num_steps = ddim_num_steps
        self.make_ddim_schedule(ddim_num_steps)
        self.make_ddim_refine_schedule(ddim_num_steps)

    def make_ddim_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps)
        betas_ddim = make_beta_schedule(schedule='linear', num_timesteps=self.num_timesteps, start=2e-4, end=4e-2, ratio=1)
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
        betas_ddim = make_beta_schedule(schedule='linear', num_timesteps=self.num_timesteps, start=2e-4, end=4e-2, ratio=1)
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
        ddim_refine_sigmas, ddim_refine_alphas, ddim_refine_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod_ddim,
                                                                                   ddim_timesteps=self.ddim_refine_timesteps,
                                                                                   eta=ddim_eta)
        self.register_buffer('ddim_refine_sigmas', ddim_refine_sigmas)
        self.register_buffer('ddim_refine_alphas', ddim_refine_alphas)
        self.register_buffer('ddim_refine_alphas_prev', ddim_refine_alphas_prev)
        self.register_buffer('ddim_refine_sqrt_one_minus_alphas', torch.sqrt(1. - ddim_refine_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def load_diffusion_net(self, net_state_dict):
        # new_states = dict()
        # for k in net_state_dict.keys():
        #     if 'layer_out' not in k and 'layer_in' not in k:
        #         new_states[k] = net_state_dict[k]
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

    def forward_t(self, l_0_batch, image, detect_box, t, reparam=False, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        batch_size = l_0_batch.shape[0]
        e = torch.randn_like(l_0_batch).to(l_0_batch.device)
        l_t_noise = q_sample(l_0_batch, self.alphas_bar_sqrt,
                             self.one_minus_alphas_bar_sqrt, t, noise=e)
        # cond c
        # l_t_input_c = l_0_batch.clone()
        # l_t_input_c[:, :, self.num_class:] = l_t_noise[:, :, self.num_class:]
        #
        # # cond cwh
        # l_t_input_cwh = l_0_batch.clone()
        # l_t_input_cwh[:, :, self.num_class:self.num_class + 2] = l_t_noise[:, :, self.num_class:self.num_class + 2]
        #
        # # cond complete
        # real_label = torch.argmax(l_0_batch[:, :, :self.num_class], dim=2)
        # real_mask = (real_label != 0).clone().detach()
        # fix_mask = rand_fix(batch_size, real_mask, ratio=0.2)
        # l_t_input_complete = l_t_noise.clone()
        # l_t_input_complete[fix_mask] = l_0_batch[fix_mask]

        # random_num = random.randint(0, 3)
        # if random_num == 1:
        #     l_t_noise = l_t_input_c
        # elif random_num == 2:
        #     l_t_noise = l_t_input_cwh
        # elif random_num == 3:
        #     l_t_noise = l_t_input_complete

        # l_t_input_all = torch.cat([l_t_noise, l_t_input_c, l_t_input_cwh, l_t_input_complete], dim=0)
        # e_all = torch.cat([e, e, e, e], dim=0)
        # t_all = torch.cat([t, t, t, t], dim=0)
        # image = torch.cat([image, image, image, image], dim=0)
        # detect_box = torch.cat([detect_box, detect_box, detect_box, detect_box], dim=0)

        eps_theta = self.model(l_t_noise, image, detect_box, timestep=t)

        return eps_theta, e

        # tt = self.sample_t_temp((batch_size,), t_max=1000)
        #
        # for i in range(1000):
        #     t = tt[:,i]
        #     l_t_noise = q_sample(l_0_batch, self.alphas_bar_sqrt,
        #                         self.one_minus_alphas_bar_sqrt, t, noise=e)
        #     bbox_1, label_1, mask_1 = self.finalize(l_t_noise, self.num_class)
        #     label_1 = label_1.unsqueeze(2)
        #     dir = '/mnt/data/ly24/result/pku/time_result/cosine_1_4/'
        #     os.makedirs(dir, exist_ok=True)
        #
        #     draw(bbox_1, label_1, imgs, i, dir, w=513, h=750)
        # sys.exit()

        # rand_mask = torch.rand(image.size(0), device=image.device)
        # mask = rand_mask < 0.1
        # image_masked = image.clone()
        # image_masked[mask] = 0
        # detect_box[mask] = 0
        # eps_theta = self.model(l_t_noise, image, detect_box, timestep=t)

        # e_t_cond, w1 = self.model(l_t_noise, image, detect_box, timestep=t)
        #
        # image_zero = torch.zeros_like(image, device=image.device, dtype=image.dtype)
        # detect_box_zero = torch.zeros_like(detect_box, device=detect_box.device, dtype=detect_box.dtype)
        # e_t_uncond, w2 = self.model(l_t_noise, image_zero, detect_box_zero, timestep=t)
        #
        # w1_exp = torch.exp(w1) / (torch.exp(w1) + torch.exp(w2))
        # w2_exp = torch.exp(w2) / (torch.exp(w1) + torch.exp(w2))
        # # w1, w2 = w[:,0], w[:,1]
        # w1_exp = w1_exp.unsqueeze(-1).expand_as(e_t_uncond)
        # w2_exp = w2_exp.unsqueeze(-1).expand_as(e_t_uncond)
        # eps_theta = w2_exp * e_t_uncond + w1_exp * e_t_cond

        # if reparam:
        #     sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, l_t_noise)
        #     sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        #     l_0_generate_reparam = 1 / sqrt_alpha_bar_t * (l_t_noise - eps_theta * sqrt_one_minus_alpha_bar_t).to(self.device)
        #
        #     return eps_theta, e, l_0_generate_reparam
        # else:
        #     return eps_theta, e

    def reverse(self, batch_size, only_last_sample=True, stochastic=True):

        self.model.eval()
        layout_t_0 = p_sample_loop(self.model, batch_size,
                                  self.num_timesteps, self.alphas,
                                  self.one_minus_alphas_bar_sqrt,
                                  only_last_sample=only_last_sample, stochastic=stochastic)

        bbox, label, mask = self.finalize(layout_t_0)

        return bbox, label, mask

    def reverse_ddim(self, image, detect_box, save_inter_dir='', img=None, stochastic=True, save_inter=False, max_len=16):
        self.model.eval()
        layout_t_0, intermediates = ddim_sample_loop(self.model, image, detect_box, self.ddim_timesteps, self.ddim_alphas,
                                                     self.ddim_alphas_prev, self.ddim_sigmas, stochastic=stochastic,
                                                     seq_len=max_len, seq_dim=self.seq_dim)
        # print(layout_t_0)
        bbox, label, mask = self.finalize(layout_t_0, self.num_class)
        if not save_inter:
            return bbox, label, mask

        else:

            for i, layout_t in enumerate(intermediates['y_inter']):

                bbox_1, label_1, mask_1 = self.finalize(layout_t, self.num_class)
                draw_single(bbox_1, label_1, img, i, save_inter_dir, w=513, h=750)

            return bbox, label, mask

    @staticmethod
    def finalize(layout, num_class):
        bbox = layout[:, :, num_class:]
        bbox = torch.clamp(bbox, min=-1, max=1) / 2 + 0.5
        label = torch.argmax(layout[:, :, :num_class], dim=2)
        mask = (label != 0).clone().detach()
        label = label.unsqueeze(-1)
        return bbox, label, mask

    def conditional_reverse_ddim(self, real_layout, image, detect_box, imgs=None, cond='c', ratio=0.2, stochastic=True, save_inter=False):

        self.model.eval()
        layout_t_0, intermediates = \
            ddim_cond_sample_loop(self.model, real_layout, image, detect_box, self.ddim_timesteps, self.ddim_alphas,
                                  self.ddim_alphas_prev, self.ddim_sigmas, stochastic=stochastic, cond=cond,
                                  ratio=ratio)

        bbox, label, mask = self.finalize(layout_t_0, self.num_class)

        if not save_inter:
            return bbox, label, mask
        else:
            dir = '/home/kl23/code/ditl/plot/cwh_2/'
            os.makedirs(dir, exist_ok=True)
            for i, layout_t in enumerate(intermediates['y_inter']):
                bbox_1, label_1, mask_1 = self.finalize(layout_t, self.num_class)
                draw_single(bbox_1, label_1, imgs[0], i, dir, w=513, h=750)

            return bbox, label, mask

    def refinement_reverse_ddim(self, noisy_layout, image, detect_box, imgs=None):
        self.model.eval()

        layout_t_0, intermediates = \
            ddim_refine_sample_loop(self.model, noisy_layout, image, detect_box, self.ddim_refine_timesteps, self.ddim_refine_alphas,
                                    self.ddim_refine_alphas_prev, self.ddim_refine_sigmas)

        bbox, label, mask = self.finalize(layout_t_0, self.num_class)

        # for i, layout_t in enumerate(intermediates['y_inter']):
        #     bbox_1, label_1, mask_1 = self.finalize(layout_t, self.num_class)
        #     dir = '/home/kl23/code/ditl/plot/refine_medium_denoise/'
        #     os.makedirs(dir, exist_ok=True)
        #
        #     draw_single(bbox_1, label_1, imgs[0], i, dir, w=513, h=750)

        return bbox, label, mask



if __name__ == "__main__":

    model = Diffusion(num_timesteps=1000, nhead=8, dim_transformer=1024,
                           feature_dim=2048, seq_dim=10, num_layers=4,
                           device='cpu', ddim_num_steps=200, embed_type='pos')

    print(pow(model.one_minus_alphas_bar_sqrt[201], 2))


    print(sum(model.ddim_timesteps <= 201))
    timesteps = model.ddim_timesteps
    total_steps = sum(model.ddim_timesteps <= 201)
    time_range = np.flip(timesteps[:total_steps])
    print(time_range)
