from data_utils import utils
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import idx2numpy

class MNISTDataset(Dataset):
    def __init__(self, img_path, label_path):
        super().__init__()

        imgs = idx2numpy.convert_from_file(img_path)
        labels = idx2numpy.convert_from_file(label_path)

        imgs = torch.tensor(imgs, dtype= torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
    
        self.__data = {}

        for idx, (img, label) in enumerate(zip(imgs, labels)):
            self.__data[idx] = {
                "image": img,
                "label": label
            }

    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, index):
        return self.__data[index]

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

    def load_train_dev(self):
        train_dataset = MNISTDataset(self.train_img_path, self.train_label_path)
        train_size = int(0.8*len(train_dataset))
        dev_size = len(train_dataset) - train_size
        train_dataset, dev_dataset = random_split(train_dataset, [train_size, dev_size])


        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True, collate_fn= utils.collate_fn)
        dev_dataloader = DataLoader(dev_dataset, self.test_batch, shuffle= True, collate_fn= utils.collate_fn)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = MNISTDataset(self.test_img_path, self.test_label_path)
        test_dataloader = DataLoader(test_dataset, self.train_batch, shuffle= False, collate_fn= utils.collate_fn)
        return test_dataloader