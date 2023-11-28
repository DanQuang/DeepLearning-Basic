import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_H = config["image_H"]
        self.image_W = config["image_W"]
        self.image_C = config["image_C"]
        self.num_classes = config["num_classes"]

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels= self.image_C, out_channels= 6, kernel_size= 5, padding= 2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size= 2, stride= 2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels= 6, out_channels= 16, kernel_size= 5, padding= 0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size= 2, stride= 2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features= 120),
            nn.Sigmoid(),
            nn.LazyLinear(out_features= 84),
            nn.Sigmoid(),
            nn.LazyLinear(out_features= self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(self.block_2(self.block_1(x)))