# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Config import args_setting
from Dataset import Speckle_Origin
from TrainValTest import train, val, test
from Models import UGenerator
import os
import tqdm
import time
import GPUtil
from math import inf
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


def main():
    args = args_setting()
    input_root = os.path.join(args.dir_path, 'input')
    output_root = os.path.join(args.dir_path, 'output')
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    train_dataset = Speckle_Origin(root=input_root, split='train', transform=True, args=args)
    val_dataset = Speckle_Origin(root=input_root, split='val', transform=True, args=args)
    test_dataset = Speckle_Origin(root=input_root, split='test', transform=True, args=args)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    train_dataset_size, val_dataset_size, test_dataset_size = len(train_dataset), len(val_dataset), len(test_dataset)
    print('Number of sequence in train dataset/ val dataset/ test dataset: {}/ {}/ {}.'.format(train_dataset_size, val_dataset_size, test_dataset_size))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = UGenerator(in_channels=args.in_channels, num_classes=args.num_classes, num_filters=args.base_c, args=args).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_save_path = os.path.join(output_root, 'reconstraction.pth')
    for epoch in tqdm.trange(args.num_epoch):
        train(model, optimizer, criterion, train_loader, device, train_dataset_size, epoch, args)
        val(model, val_loader, device, val_dataset_size, model_save_path, args)
    test(model, test_loader, device, test_dataset_size, model_save_path, args)


if __name__ == '__main__':
    main()