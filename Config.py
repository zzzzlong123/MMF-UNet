# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
import argparse


def args_setting():
    parser = argparse.ArgumentParser(description='OrdinaryUnet')
    parser.add_argument('--dir_path', type=str, default='G:/')
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--speckle_size', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--origin_size', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_type', type=str, default='UGenerator')
    parser.add_argument('--base_c', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4, help='选择learning rate')
    parser.add_argument('--L', type=int, default=255)
    args = parser.parse_args()
    return args
