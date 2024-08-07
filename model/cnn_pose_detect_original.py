# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:44:31 2020

@author: MrHossein
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Pos_D(nn.Module):
    def __init__(self):
        super(CNN_Pos_D, self).__init__()
        # input size = 220*220*3
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=4)  # output: 55*55*96
        self.LRN1 = nn.LocalResponseNorm(5)  # output: 55*55*96
        self.MaxP1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # output: 27*27*96

        self.Conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1,
                               padding=2)  # output: 27*27*256
        self.LRN2 = nn.LocalResponseNorm(5)  # output: 27*27*256
        self.MaxP2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # output: 13*13*256

        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1,
                               padding=1)  # output: 13*13*384

        self.Conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1,
                               padding=1)  # output: 13*13*384
        self.Conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1,
                               padding=1)  # output: 13*13*256



        self.MaxP3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # output: 6*6*256
        self.Dpout0 = nn.Dropout2d(p=0.6)
        self.FC1 = nn.Linear(9216, 4096)
        self.FC2 = nn.Linear(4096, 4096)
        self.FC3 = nn.Linear(4096, 28)

    def forward(self, X):
        X = self.Conv1(X)
        X = F.relu(X)
        X = self.LRN1(X)
        X = self.MaxP1(X)

        X = self.Conv2(X)
        X = F.relu(X)
        X = self.LRN2(X)
        X = self.MaxP2(X)

        X = self.Conv3(X)
        X = F.relu(X)

        X = self.Conv4(X)
        X = F.relu(X)

        X = self.Conv5(X)
        X = F.relu(X)
        X = self.MaxP3(X)

        X = self.Dpout0(X)
        X = torch.flatten(X, 1)
        X = self.FC1(X)
        X = F.relu(X)

        X = self.FC2(X)
        X = F.relu(X)

        output = self.FC3(X)

        return output