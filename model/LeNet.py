import torch
from torch import nn

# Custom LeNet using ReLU
class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        self.block_1 = nn.Sequential(
            nn.LazyConv2d(out_channels= 6, kernel_size= 5, padding= 2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size= 2, stride= 2)
        )

        self.block_2 = nn.Sequential(
            nn.LazyConv2d(out_channels= 16, kernel_size= 5, padding= 0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size= 2, stride= 2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features= 120),
            nn.ReLU(),
            nn.LazyLinear(out_features= 84),
            nn.ReLU(),
            nn.LazyLinear(out_features= self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.dense(self.block_2(self.block_1(x)))