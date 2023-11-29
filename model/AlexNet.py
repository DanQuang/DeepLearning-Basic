import torch
from torch import nn

# Custom LeNet using ReLU
class AlexNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_H = config["image_H"]
        self.image_W = config["image_W"]
        self.image_C = config["image_C"]
        self.num_classes = config["num_classes"]

        self.block_1 = nn.Sequential(
            nn.LazyConv2d(out_channels= 96, kernel_size= 11, stride= 4, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.block_2 = nn.Sequential(
            nn.LazyConv2d(out_channels= 256, kernel_size= 5, padding= 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.block_3 = nn.Sequential(
            nn.LazyConv2d(out_channels= 384, kernel_size= 3, padding= 1), nn.ReLU(),
            nn.LazyConv2d(out_channels= 384, kernel_size= 3, padding= 1), nn.ReLU(),
            nn.LazyConv2d(out_channels= 256, kernel_size= 3, padding= 1), nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features= 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(out_features= 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(out_features= self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.dense(self.block_3(self.block_2(self.block_1(x))))