import torch
from torch import nn

# Custom AlexNet for small size (28x28)
class AlexNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        # Custom block_1
        self.block_1 = nn.Sequential(
            nn.LazyConv2d(out_channels= 96, kernel_size= 11, stride= 5, padding= 1), # kernel_size = 11, stride = 5
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 5, stride= 2) # kernel_size= 5
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