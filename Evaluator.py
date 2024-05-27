# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2023.07.18
"""
import torch


def ssim(preds, origins, L, batch_size, device):
    L = torch.tensor(L).to(device).float()
    batch_size_tensor = torch.tensor(batch_size).to(device).float()
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    structural_similarity_index = torch.tensor(0).to(device).float()
    for iiii in range(batch_size):
        mu1 = torch.mean(preds[iiii,:,:,:])
        mu2 = torch.mean(origins[iiii,:,:,:])
        mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq, sigma2_sq = torch.var(preds[iiii,:,:,:]), torch.var(origins[iiii,:,:,:])
        sigma12 = torch.mean(preds[iiii,:,:,:] * origins[iiii,:,:,:]) - mu1_mu2
        structural_similarity_index += ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq * mu2_sq + C1) * (sigma1_sq * sigma2_sq + C2))
    return structural_similarity_index/batch_size_tensor


def psnr(preds, origins, L, batch_size, device):
    L = torch.tensor(L).to(device).float()
    batch_size_tensor = torch.tensor(batch_size).to(device).float()
    peak_signal_to_noise_ratio = torch.tensor(0).to(device).float()
    for iiii in range(batch_size):
        mse = torch.mean((preds[iiii,:,:,:] - origins[iiii,:,:,:]) ** 2)
        if mse == 0:
            return float('inf')
        peak_signal_to_noise_ratio += 10 * torch.log10(L*L/torch.mean((preds[iiii,:,:,:] - origins[iiii,:,:,:]) ** 2))
    return peak_signal_to_noise_ratio/batch_size_tensor


def mse(preds, origins, L, batch_size, device):
    L = torch.tensor(L).to(device).float()
    batch_size_tensor = torch.tensor(batch_size).to(device).float()
    mean_square_error = torch.tensor(0).to(device).float()
    for iiii in range(batch_size):
        mean_square_error += torch.mean((preds[iiii,:,:,:] - origins[iiii,:,:,:]) ** 2)
    return mean_square_error / batch_size_tensor
