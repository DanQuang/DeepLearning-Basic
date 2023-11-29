from data_utils import MNISTDataset, utils
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms

def unsqueeze(x: torch.Tensor) ->torch.Tensor:
    return x.unsqueeze(0)

class Load_Data:
    def __init__(self, config):
        # train
        self.train_img_path = config["train_img_path"]
        self.train_label_path = config["train_label_path"]
        # test
        self.test_img_path = config["test_img_path"]
        self.test_label_path = config["test_label_path"]

        self.train_batch = config["train_batch"]
        self.dev_batch = config["dev_batch"]
        self.test_batch = config["test_batch"]

        self.image_H = config["image_H"]
        self.image_W = config["image_W"]
        self.image_C = config["image_C"]

        self.transforms = transforms.Compose([
            transforms.Lambda(unsqueeze),
            transforms.Resize((self.image_H, self.image_W), antialias= True),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def load_train_dev(self):
        train_dataset = MNISTDataset.MNISTDataset(self.transforms, self.train_img_path, self.train_label_path)
        idx = int(0.8*len(train_dataset))
        dev_dataset = train_dataset[idx:]
        train_dataset = train_dataset[:idx]

        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True, collate_fn= utils.collate_fn)
        dev_dataloader = DataLoader(dev_dataset, self.test_batch, shuffle= True, collate_fn= utils.collate_fn)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = MNISTDataset.MNISTDataset(self.transforms, self.test_img_path, self.test_label_path)
        test_dataloader = DataLoader(test_dataset, self.train_batch, shuffle= False, collate_fn= utils.collate_fn)
        return test_dataloader