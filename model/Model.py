from torch import nn 
import torch
from model import AlexNet, GoogLeNet, LeNet, NiN, VGG

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config["model"] == "LeNet":
            self.model = LeNet.LeNet(config)
        if config["model"] == "AlexNet":
            self.model = AlexNet.AlexNet(config)
        if config["model"] == "VGG":
            self.model = VGG.VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), config= config)
        if config["model"] == "NiN":
            self.model = NiN.NiN(config)
        if config["model"] == "GoogLeNet":
            self.model = GoogLeNet.GoogLeNet(config)

    def forward(self, x):
        return self.model(x)