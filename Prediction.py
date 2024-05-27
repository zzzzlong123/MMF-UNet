# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2023.07.18
"""
from Config import args_setting
from Models import UGenerator
from Dataset import Speckle_Origin
from Evaluator import ssim, psnr, mse
import os
import shutil
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

args = args_setting()
input_root = os.path.join(args.dir_path, 'input')
output_root = os.path.join(args.dir_path, 'output')
if os.path.exists(os.path.join(output_root, 'prediction')):
    shutil.rmtree(os.path.join(output_root, 'prediction'))
os.makedirs(os.path.join(output_root, 'prediction'))
prediction_dataset = Speckle_Origin(root=input_root, split='prediction', transform=True, args=args)
prediction_loader = DataLoader(dataset=prediction_dataset, batch_size=args.batch_size, shuffle=None, num_workers=0, drop_last=False)
prediction_dataset_size = len(prediction_dataset)
print('Number of sequence in prediction dataset: {}.'.format(prediction_dataset_size))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UGenerator(in_channels=args.in_channels, num_classes=args.num_classes, num_filters=args.base_c,
                   args=args).to(device)
criterion = nn.MSELoss().to(device)
model.eval()
ssim_sum, mse_sum, psnr_sum = 0, 0, 0
model_save_path = os.path.join(output_root, 'reconstraction.pth')
model.load_state_dict(torch.load(model_save_path))
with torch.no_grad():
    for batch_idx, (speckles, origins) in enumerate(prediction_loader):
        speckles, origins = speckles.to(device), origins.to(device)
        preds = model(speckles)
        loss = criterion(preds, origins)
        print('MSE of every batch is {}'.format(loss.item()))
        ssim_sum += ssim(preds, origins, args.L, args.batch_size, device).item()
        psnr_sum += psnr(preds, origins, args.L, args.batch_size, device).item()
        mse_sum += mse(preds, origins, args.L, args.batch_size, device).item()
    ssim_average = ssim_sum / (batch_idx+1)
    mse_average = mse_sum / (batch_idx+1)
    psnr_average = psnr_sum / (batch_idx+1)
    print('prediction results: \nSSIM_average: ', ssim_average, " MSE_average: ", mse_average, " PSNR_average: ",
          psnr_average)
