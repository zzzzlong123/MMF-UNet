# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Evaluator import ssim, psnr, mse
import torch


def train(model, optimizer, criterion, train_loader, device, train_dataset_size, epoch, args):
    model.train()
    ssim_sum, mse_sum, psnr_sum = 0, 0, 0
    for batch_idx, (speckles, origins) in enumerate(train_loader):
        optimizer.zero_grad()
        speckles, origins = speckles.to(device), origins.to(device)
        preds = model(speckles)
        train_loss = criterion(preds, origins)
        train_loss.backward()
        optimizer.step()
        ssim_sum += ssim(preds, origins, args.L, args.batch_size, device).item()
        psnr_sum += psnr(preds, origins, args.L, args.batch_size, device).item()
        mse_sum += mse(preds, origins, args.L, args.batch_size, device).item()
    ssim_average = ssim_sum / int(train_dataset_size / args.batch_size)
    mse_average = mse_sum / int(train_dataset_size / args.batch_size)
    psnr_average = psnr_sum / int(train_dataset_size / args.batch_size)
    print('\ntrain results: \nSSIM_average: ', ssim_average, " MSE_average: ", mse_average, " PSNR_average: ",
          psnr_average)
    if epoch == 0:
        print("model is on GPU!" if next(model.parameters()).is_cuda else "model is on CPU!")
        print("speckles is on GPU!" if speckles.is_cuda else "speckles is on CPU!")
        print("origins is on GPU!" if origins.is_cuda else "origins is on CPU!")
        print("train_loss is on GPU!" if train_loss.is_cuda else "train_loss is on CPU!")


def val(model, val_loader, device, val_dataset_size, model_save_path, args):
    model.eval()
    ssim_sum, mse_sum, psnr_sum = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (speckles, origins) in enumerate(val_loader):
            speckles, origins = speckles.to(device), origins.to(device)
            preds = model(speckles)
            ssim_sum += ssim(preds, origins, args.L, args.batch_size, device).item()
            psnr_sum += psnr(preds, origins, args.L, args.batch_size, device).item()
            mse_sum += mse(preds, origins, args.L, args.batch_size, device).item()
        ssim_average = ssim_sum / int(val_dataset_size / args.batch_size)
        mse_average = mse_sum / int(val_dataset_size / args.batch_size)
        psnr_average = psnr_sum / int(val_dataset_size / args.batch_size)
        print('val results: \nSSIM_average: ', ssim_average, " MSE_average: ", mse_average, " PSNR_average: ",
              psnr_average)
        torch.save(model.state_dict(), model_save_path)


def test(model, test_loader, device, test_dataset_size, model_save_path, args):
    model.eval()
    ssim_sum, mse_sum, psnr_sum = 0, 0, 0
    model.load_state_dict(torch.load(model_save_path))
    with torch.no_grad():
        for batch_idx, (speckles, origins) in enumerate(test_loader):
            speckles, origins = speckles.to(device), origins.to(device)
            preds = model(speckles)
            ssim_sum += ssim(preds, origins, args.L, args.batch_size, device).item()
            psnr_sum += psnr(preds, origins, args.L, args.batch_size, device).item()
            mse_sum += mse(preds, origins, args.L, args.batch_size, device).item()
        ssim_average = ssim_sum / int(test_dataset_size / args.batch_size)
        mse_average = mse_sum / int(test_dataset_size / args.batch_size)
        psnr_average = psnr_sum / int(test_dataset_size / args.batch_size)
        print('test results: \nSSIM_average: ', ssim_average, " MSE_average: ", mse_average, " PSNR_average: ",
              psnr_average)
