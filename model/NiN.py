import torch
from torch import nn

def nin_block(num_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.LazyConv2d(num_channels, kernel_size, stride, padding), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size= 1), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size= 1), nn.ReLU()
    )

class NiN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.net = nn.Sequential(
            nin_block(num_channels= 96, kernel_size= 11, stride= 4, padding= 0),
            nn.MaxPool2d(3, stride= 2),
            nin_block(num_channels= 256, kernel_size= 5, stride= 1, padding= 2),
            nn.MaxPool2d(3, stride= 2),
            nin_block(num_channels= 384, kernel_size= 3, stride= 1, padding= 1),
            nn.MaxPool2d(3, stride= 2),
            nn.Dropout(0.5),
            nin_block(num_channels= self.num_classes, kernel_size= 3, stride= 1, padding= 1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.net(x)