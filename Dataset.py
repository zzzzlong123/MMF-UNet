# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Config import args_setting
import re
import os
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class Speckle_Origin(Dataset):
    def __init__(self, root, split='train', transform=False, args=args_setting):
        self.root = root
        self.transform = transform
        self.split = split
        self.args = args
        assert self.split in ['train', 'val', 'test', 'prediction']
        self.speckle_path = os.path.join(root, 'speckle')
        self.origin_path = os.path.join(root, 'origin')
        self.speckle_list = [f for f in os.listdir(self.speckle_path) if f.endswith('.png') or f.endswith('.jpg')]
        self.origin_list = [f for f in os.listdir(self.origin_path) if f.endswith('.png') or f.endswith('.jpg')]
        assert len(self.speckle_list) == len(self.origin_list)
        random.seed(self.args.seed)
        random.shuffle(self.speckle_list)
        random.shuffle(self.origin_list)
        if self.split == 'train':
            self.speckle_list = self.speckle_list[:int(args.train_ratio * len(self.speckle_list))]
            self.origin_list = self.origin_list[:int(args.train_ratio * len(self.origin_list1))]
        elif self.split == 'val':
            self.speckle_list = self.speckle_list[int(args.train_ratio * len(self.speckle_list)):int((args.train_ratio+args.val_ratio) * len(self.speckle_list))]
            self.origin_list = self.origin_list[int(args.train_ratio * len(self.origin_list1)):int((args.train_ratio+args.val_ratio) * len(self.origin_list))]
        elif self.split == 'test':
            self.speckle_list = self.speckle_list[int((args.train_ratio + args.val_ratio) * len(self.speckle_list)):]
            self.origin_list = self.origin_list[int((args.train_ratio + args.val_ratio) * len(self.origin_list)):]
        self.samples_speckle = [(os.path.join(self.speckle_path, filename), 0) for filename in self.speckle_list]
        self.samples_origin = [(os.path.join(self.origin_path, filename), 0) for filename in self.origin_list]

    def _len__(self):
        return len(self.samples_speckle)

    def __getitem__(self, idx):
        path_speckle, _ = self.samples_speckle[idx]
        path_origin, _ = self.samples_origin[idx]
        speckle = default_loader(path_speckle)
        origin = default_loader(path_origin)
        speckle_transform = transforms.Compose([transforms.Resize((self.args.speckle_size, self.args.speckle_size)),
                                                    transforms.Grayscale(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (0.5,))])
        origin_transform = transforms.Compose([transforms.Resize((self.args.origin_size, self.args.origin_size)),
                                                 transforms.Grayscale(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5,), (0.5,))])

        if self.transform is not False:
            speckle = speckle_transform(speckle)
            origin = origin_transform(origin)

        return speckle, origin
