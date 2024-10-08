import sys

import math
import einops
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import logger

def make_beta_schedule(schedule="linear", num_timesteps=1000, start=2e-4, end=4e-2, ratio=1):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps) / ratio
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min((1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2)) / ratio, max_beta)
             for i in
             range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def rand_fix(batch_size, mask, ratio=0.2, n_elements=16, stochastic=True):
    if stochastic:
        indices = (torch.rand([batch_size, n_elements]) <= torch.rand([1]).item() * ratio).to(mask.device) * mask.to(torch.bool)
    else:
        a = torch.tensor([True,
                          False, False, False, False, False, False, False, False, False,]).to(mask.device)
        indices = einops.repeat(a, "l -> n l", n=batch_size) * mask.to(torch.bool)
    return indices

# Forward functions
def q_sample(y, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, fix_mask, t, noise=None, cond='uncond'):
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    if cond == 'uncond':
        y_t = sqrt_alpha_bar_t * y + sqrt_one_minus_alpha_bar_t * noise
    else:
        y_t_noise = (sqrt_alpha_bar_t * y
                     + sqrt_one_minus_alpha_bar_t * noise) * (~fix_mask)
        y_t_nonoise = y * (fix_mask)

        y_t = y_t_noise + y_t_nonoise
    return y_t


# Reverse function -- sample y_{t-1} given y_t
def p_sample(model, y_t, t, alphas, one_minus_alphas_bar_sqrt, stochastic=True):
    """
    Reverse diffusion process sampling -- one time step.
    y: sampled y at time step t, y_t.
    """
    device = next(model.parameters()).device
    z = stochastic * torch.randn_like(y_t)
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y_t)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y_t)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())

    eps_theta = model(y_t, timestep=t).to(device).detach()

    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (y_t - eps_theta * sqrt_one_minus_alpha_bar_t).to(device)

    # posterior mean
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y_t

    # posterior variance
    beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)

    return y_t_m_1


# Reverse function -- sample y_0 given y_1
def p_sample_t_1to0(model, y_t, one_minus_alphas_bar_sqrt):
    device = next(model.parameters()).device
    t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = model(y_t, timestep=t).to(device).detach()

    # y_0 reparameterization
    y_0_reparam = 1 / sqrt_alpha_bar_t * (y_t - eps_theta * sqrt_one_minus_alpha_bar_t).to(device)

    y_t_m_1 = y_0_reparam.to(device)

    return y_t_m_1


def p_sample_loop(model, batch_size, n_steps, alphas, one_minus_alphas_bar_sqrt,
                  only_last_sample=True, stochastic=True):
    num_t, l_p_seq = None, None

    device = next(model.parameters()).device

    l_t = stochastic * torch.randn_like(torch.zeros([batch_size, 25, 10])).to(device)
    if only_last_sample:
        num_t = 1
    else:
        # y_p_seq = [y_t]
        l_p_seq = torch.zeros([batch_size, 25, 10, n_steps + 1]).to(device)
        l_p_seq[:, :, :, n_steps] = l_t

    for t in reversed(range(1, n_steps - 1)):

        l_t = p_sample(model, l_t, t, alphas, one_minus_alphas_bar_sqrt, stochastic=stochastic)  # y_{t-1}

        if only_last_sample:
            num_t += 1
        else:
            # y_p_seq.append(y_t)
            l_p_seq[:, :, :, t] = l_t

    if only_last_sample:
        l_0 = p_sample_t_1to0(model, l_t, one_minus_alphas_bar_sqrt)
        return l_0
    else:
        # assert len(y_p_seq) == n_steps
        l_0 = p_sample_t_1to0(model, l_p_seq[:, :, :, 1], one_minus_alphas_bar_sqrt)
        # y_p_seq.append(y_0)
        l_p_seq[:, :, :, 0] = l_0

        return l_0, l_p_seq


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    elif ddim_discr_method == 'new':
        c = (num_ddpm_timesteps - 50) // (num_ddim_timesteps - 50)
        ddim_timesteps = np.asarray(list(range(0, 50)) + list(range(50, num_ddpm_timesteps - 50, c)))
    elif ddim_discr_method == 'refine':
        ddim_timesteps = np.asarray(list(range(0, 1000, 2)))
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    steps_out = ddim_timesteps
    # steps_out = ((1 - (steps_out / num_ddpm_timesteps)**3) * num_ddpm_timesteps).astype(int)
    # steps_out = np.flip(steps_out)

    # print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta):
    torch.cuda.synchronize()
    # select alphas for computing the variance schedule
    device = alphacums.device
    ddim_timesteps = ddim_timesteps.copy()
    if ddim_timesteps.shape[0] == 1000:
        alphas = alphacums[ddim_timesteps]
        alphas_prev = torch.cat([torch.ones(1).to(device), alphacums[:-1]], dim=0)
    else:
        # alphas = alphacums[ddim_timesteps]
        alphas = alphacums[ddim_timesteps]
        alphas_prev = torch.tensor([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist()).to(device)

    sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    # print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
    # print(f'For the chosen value of eta, which is {eta}, '
    #       f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def ddim_sample_loop(model, image, sal_box, timesteps, ddim_alphas, ddim_alphas_prev,
                     ddim_sigmas, seq_len=16, seq_dim=8):
    device = next(model.parameters()).device
    batch_size = image.shape[0]
    b_t = 1  * torch.randn_like(torch.zeros([batch_size, seq_len, seq_dim])).to(device)

    intermediates = {'y_inter': [b_t], 'pred_y0': [b_t]}
    time_range = np.flip(timesteps)

    total_steps = timesteps.shape[0]
    # print(f"Running DDIM Sampling with {total_steps} timesteps")
    for i, step in enumerate(time_range):
        index = total_steps - i - 1
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        b_t, pred_y0 = ddim_sample_step(model, b_t, image, sal_box, t, index, ddim_alphas,
                                        ddim_alphas_prev, ddim_sigmas)
        intermediates['y_inter'].append(b_t)
        intermediates['pred_y0'].append(pred_y0)
    return b_t, intermediates


def ddim_cond_sample_loop(model, real_layout, image, sal_box, timesteps,
                          ddim_alphas, ddim_alphas_prev, ddim_sigmas,
                          stochastic=True, cond='c', ratio=0.1):
    device = next(model.parameters()).device
    batch_size, seq_len, seq_dim = real_layout.shape
    num_class = seq_dim - 4
    real_label = torch.argmax(real_layout[:, :, :num_class], dim=2)
    real_mask = (real_label != 0).clone().detach()

    if cond == 'complete':
        fix_mask = rand_fix(batch_size, real_mask, ratio=ratio, n_elements=seq_len, stochastic=stochastic)
    elif cond == 'c':
        fix_mask = torch.zeros([batch_size, seq_len, seq_dim]).to(torch.bool)
        fix_mask[:, :, :num_class] = True
    elif cond == 'cwh':
        fix_mask = torch.zeros([batch_size, seq_len, seq_dim]).to(torch.bool)
        fix_ind = [x for x in range(num_class)] + [num_class + 2, num_class + 3]
        fix_mask[:, :, fix_ind] = True
    else:
        raise Exception('cond must be c, cwh, or complete')

    l_t = 1 * torch.randn_like(torch.zeros([batch_size, seq_len, seq_dim])).to(device)
    intermediates = {'y_inter': [l_t], 'pred_y0': [l_t]}
    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]

    l_t[fix_mask] = real_layout[fix_mask]
    fix_mask = fix_mask.to(device)

    for i, step in enumerate(time_range):
        index = total_steps - i - 1
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        # l_t[fix_mask] = real_layout[fix_mask]
        l_t, pred_y0, = ddim_sample_step(model, l_t, image, sal_box, t, index, ddim_alphas,
                                               ddim_alphas_prev, ddim_sigmas)
        l_t[fix_mask] = real_layout[fix_mask]

        intermediates['y_inter'].append(l_t)
        intermediates['pred_y0'].append(pred_y0)

    return l_t, intermediates


def ddim_refine_sample_loop(model, noisy_layout, image, sal_box, timesteps, ddim_alphas, ddim_alphas_prev,
                            ddim_sigmas):
    device = next(model.parameters()).device
    batch_size, seq_len, seq_dim = noisy_layout.shape
    l_t = noisy_layout

    intermediates = {'y_inter': [l_t], 'pred_y0': [l_t]}
    total_steps = sum(timesteps <= 10) # Depends on the degree of disturbance
    time_range = np.flip(timesteps[:total_steps])

    for i, step in enumerate(time_range):
        index = total_steps - i - 1
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        l_t, pred_y0 = ddim_sample_step(model, l_t, image, sal_box, t, index, ddim_alphas,
                                        ddim_alphas_prev, ddim_sigmas)
        intermediates['y_inter'].append(l_t)
        intermediates['pred_y0'].append(pred_y0)

    return l_t, intermediates


def ddim_sample_step(model, l_t, image, sal_box, t, index, ddim_alphas, ddim_alphas_prev, ddim_sigmas):
    device = next(model.parameters()).device

    e_t = model(l_t, image, sal_box, timestep=t)
    e_t = e_t.to(device).detach()

    # cfg
    # w = 0.6
    # image_zero = torch.zeros_like(image, device=image.device, dtype=image.dtype)
    # label_box_zero = torch.zeros_like(label_box, device=label_box.device, dtype=label_box.dtype)
    # e_t_uncond = model(l_t, image_zero, label_box_zero, timestep=t).to(device).detach()
    # e_t = e_t_uncond + w * (e_t_cond - e_t_uncond)

    sqrt_one_minus_alphas = torch.sqrt(1. - ddim_alphas)
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full(e_t.shape, ddim_alphas[index], device=device)
    a_t_m_1 = torch.full(e_t.shape, ddim_alphas_prev[index], device=device)
    sigma_t = torch.full(e_t.shape, ddim_sigmas[index], device=device)
    sqrt_one_minus_at = torch.full(e_t.shape, sqrt_one_minus_alphas[index], device=device)
    # direction pointing to x_t
    dir_b_t = (1. - a_t_m_1 - sigma_t ** 2).sqrt() * e_t
    noise = sigma_t * torch.randn_like(l_t).to(device)
    # reparameterize x_0
    b_0_reparam = (l_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
    b_t_m_1 = a_t_m_1.sqrt() * b_0_reparam + 1 * dir_b_t + noise

    return b_t_m_1, b_0_reparam

