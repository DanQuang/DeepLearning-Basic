import torch
from torch import nn

def vgg_block(num_convs, out_channels):
    layers= []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size= 3, padding= 1))
    layers.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
    return nn.Sequential(*layers)


# Custom VGGNet for small size (28x28)
class VGG(nn.Module):
    def __init__(self, arch, config):
        super().__init__()
        self.num_classes = config["num_classes"]

        convs_blks = []
        for (num_convs, out_channels) in arch:
            convs_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential(
            *convs_blks, nn.Flatten(),
            nn.LazyLinear(out_features= 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(out_features= 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(out_features= self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (x.dim() == 3):
            x = x.unsqueeze(dim = 1)
        return self.net(x)