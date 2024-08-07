import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Pos_D(nn.Module):
    def __init__(self):
        super(CNN_Pos_D, self).__init__()
        # Input size = 220*220*3
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
                               stride=4, padding=4)  # Output: 55*55*96
        self.LRN1 = nn.LocalResponseNorm(5)  # Output: 55*55*96
        self.MaxP1 = nn.MaxPool2d(kernel_size=3, stride=2,
                                  padding=0)  # Output: 27*27*96

        self.Conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5,
                               stride=1, padding=2)  # Output: 27*27*256
        self.LRN2 = nn.LocalResponseNorm(5)  # Output: 27*27*256
        self.MaxP2 = nn.MaxPool2d(kernel_size=3, stride=2,
                                  padding=0)  # Output: 13*13*256

        self.Conv3 = nn.Conv2d(in_channels=256, out_channels=384,
                               kernel_size=3, stride=1,
                               padding=1)  # Output: 13*13*384
        self.Conv4 = nn.Conv2d(in_channels=384, out_channels=384,
                               kernel_size=3, stride=1,
                               padding=1)  # Output: 13*13*384

        # Adding an extra convolutional layer with 256 output channels
        self.Conv5 = nn.Conv2d(in_channels=384, out_channels=256,
                               kernel_size=3, stride=1,
                               padding=1)  # Output: 13*13*256
        self.LRN5 = nn.LocalResponseNorm(5)  # Adding LRN to Conv5

        self.Conv6 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=3, stride=1,
                               padding=1)  # Output: 13*13*256
        self.MaxP3 = nn.MaxPool2d(kernel_size=3, stride=2,
                                  padding=0)  # Output: 6*6*256

        self.Dpout0 = nn.Dropout2d(p=0.6)
        self.FC1 = nn.Linear(6 * 6 * 256, 4096)
        self.FC2 = nn.Linear(4096, 4096)
        self.FC3 = nn.Linear(4096, 4096)
        self.FC4 = nn.Linear(4096, 28)

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
        X = self.LRN5(X)

        X = self.Conv6(X)
        X = F.relu(X)
        X = self.MaxP3(X)

        X = self.Dpout0(X)
        X = torch.flatten(X, 1)
        X = self.FC1(X)
        X = F.relu(X)

        X = self.FC2(X)
        X = F.relu(X)

        X = self.FC3(X)
        X = F.relu(X)

        output = self.FC4(X)

        return output
