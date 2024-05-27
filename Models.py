# -*- coding:utf-8 -*-

"""
author: Hailong Zhang
date: 2024.05.27
"""
from Config import args_setting
import torch
import torch.nn as nn

args = args_setting()


class UGenerator(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, num_filters=64, args=args):
        super(UGenerator, self).__init__()
        self.in_conv = nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(num_filters, 2 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(4 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.conv_down = nn.Conv2d(8 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv1 = nn.ConvTranspose2d(8 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv_up = nn.ConvTranspose2d(16 * num_filters, 8 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose2d(16 * num_filters, 4 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv3 = nn.ConvTranspose2d(8 * num_filters, 2 * num_filters, kernel_size=4, stride=2, padding=1)
        self.tconv4 = nn.ConvTranspose2d(4 * num_filters, num_filters, kernel_size=4, stride=2, padding=1)
        self.out_tconv = nn.ConvTranspose2d(2 * num_filters, num_classes, kernel_size=4, stride=2, padding=1)
        self.ins_norm0 = nn.InstanceNorm2d(num_filters)
        self.ins_norm1 = nn.InstanceNorm2d(2 * num_filters)
        self.ins_norm2 = nn.InstanceNorm2d(4 * num_filters)
        self.ins_norm3 = nn.InstanceNorm2d(8 * num_filters)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout2d(p=0.5)
        self.args = args

    def forward(self, x):
        x1 = self.in_conv(x)
        x1 = self.leakyrelu(x1)
        x2 = self.conv1(x1)
        x2 = self.ins_norm1(x2)
        x2 = self.leakyrelu(x2)
        x3 = self.conv2(x2)
        x3 = self.ins_norm2(x3)
        x3 = self.leakyrelu(x3)
        x4 = self.conv3(x3)
        x4 = self.ins_norm3(x4)
        x4 = self.leakyrelu(x4)
        x5 = self.conv_down(x4)
        x5 = self.ins_norm3(x5)
        x5 = self.leakyrelu(x5)
        x6 = self.conv_down(x5)
        x6 = self.ins_norm3(x6)
        x6 = self.leakyrelu(x6)
        x7 = self.conv_down(x6)
        x7 = self.ins_norm3(x7)
        x7 = self.leakyrelu(x7)
        x = self.conv_down(x7)
        x = self.relu(x)
        x = self.tconv1(x)
        x = self.ins_norm3(x)
        x = self.dropout(x)
        x = torch.cat([x, x7], dim=1)
        x = self.relu(x)
        x = self.tconv_up(x)
        x = self.ins_norm3(x)
        x = self.dropout(x)
        x = torch.cat([x, x6], dim=1)
        x = self.relu(x)
        x = self.tconv_up(x)
        x = self.ins_norm3(x)
        x = self.dropout(x)
        x = torch.cat([x, x5], dim=1)
        x = self.relu(x)
        x = self.tconv_up(x)
        x = self.ins_norm3(x)
        x = torch.cat([x, x4], dim=1)
        x = self.relu(x)
        x = self.tconv2(x)
        x = self.ins_norm2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.relu(x)
        x = self.tconv3(x)
        x = self.ins_norm1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.relu(x)
        x = self.tconv4(x)
        x = self.ins_norm0(x)
        x = torch.cat([x, x1], dim=1)
        x = self.relu(x)
        x = self.out_tconv(x)
        x = self.tanh(x)
        return x