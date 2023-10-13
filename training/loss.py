# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import numpy as np

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None, **kwargs):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None, **kwargs):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


@persistence.persistent_class
class NoisyEDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, epsilon=1e-3):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.epsilon = epsilon

    def __call__(self, net, images, labels=None, augment_pipe=None, **kwargs):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        noisy_image = y + n
        
        cat_input = torch.cat([noisy_image, torch.ones_like(noisy_image)], axis=1)
        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :3]

        epsilon = self.epsilon
        perturbation_noise = torch.randn_like(noisy_image)
        
        perturbed_image = noisy_image + epsilon * perturbation_noise
        cat_input = torch.cat([perturbed_image, torch.ones_like(perturbed_image)], axis=1)
        D_yn_perturbed = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :3]

        div_loss = (((D_yn_perturbed - D_yn) / epsilon).reshape(D_yn.shape[0], -1) * perturbation_noise.reshape(D_yn.shape[0], -1)).sum(axis=1)
        identity_loss = ((D_yn - noisy_image) ** 2).reshape(D_yn.shape[0], -1).sum(axis=1)
        loss = 2 * weight.squeeze() * div_loss + identity_loss
        return loss, loss, loss

#----------------------------------------------------------------------------
# EDMLoss for Ambient Diffusion

@persistence.persistent_class
class AmbientLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, norm=2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.norm = norm

    def __call__(self, net, images, corruption_matrix, hat_corruption_matrix, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        
        masked_image = hat_corruption_matrix * (y + n)
        noisy_image = masked_image

        cat_input = torch.cat([noisy_image, hat_corruption_matrix], axis=1)
        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :3]
        
        if self.norm == 2:
            train_loss = weight * ((hat_corruption_matrix * (D_yn - y)) ** 2)
            val_loss = weight * ((corruption_matrix * (D_yn - y)) ** 2)
            test_loss = weight * ((D_yn - y) ** 2)
        elif self.norm == 1:
            # l1 loss
            train_loss = weight * (hat_corruption_matrix * torch.abs(D_yn - y))
            val_loss = weight * (corruption_matrix * torch.abs(D_yn - y))
            test_loss = weight * torch.abs(D_yn - y)
        else:
            # raise exception
            raise ValueError("Wrong norm type. Use 1 or 2.")
        return train_loss, val_loss, test_loss
#----------------------------------------------------------------------------
# VPLoss for Ambient Diffusion
@persistence.persistent_class
class AmbientVPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5, norm=2):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.norm = norm

    def __call__(self, net, images, corruption_matrix, hat_corruption_matrix, labels, augment_pipe=None, **kwargs):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        cat_input = torch.cat([hat_corruption_matrix * (y + n), hat_corruption_matrix], axis=1)
        D_yn = net(cat_input, sigma, labels, augment_labels=augment_labels)[:, :3]

        if self.norm == 2:
            train_loss = weight * ((hat_corruption_matrix * (D_yn - y)) ** 2)
            val_loss = weight * ((corruption_matrix * (D_yn - y)) ** 2)
            test_loss = weight * ((D_yn - y) ** 2)
        elif self.norm == 1:
            # l1 loss
            train_loss = weight * (hat_corruption_matrix * torch.abs(D_yn - y))
            val_loss = weight * (corruption_matrix * torch.abs(D_yn - y))
            test_loss = weight * torch.abs(D_yn - y)
        else:
            # raise exception
            raise ValueError("Wrong norm type. Use 1 or 2.")
        return train_loss, val_loss, test_loss


    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
