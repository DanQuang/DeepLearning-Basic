from data_utils import utils
from torch.utils.data import DataLoader, Dataset
import torch
import idx2numpy

class MNISTDataset(Dataset):
    def __init__(self, img_path, label_path):
        super().__init__()

        imgs = idx2numpy.convert_from_file(img_path)
        labels = idx2numpy.convert_from_file(label_path)

        imgs = torch.tensor(imgs, dtype= torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
    
        self.__data = [{"image": img ,"label": label} for img, label in zip(imgs, labels)]

    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.start or 0, index.stop or len(self), index.step or 1
            selected_data = [{"image": self.__data[i]["image"], "label": self.__data[i]["label"]} for i in range(start, stop, step)]
            return selected_data
        else:
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
        idx = int(0.8*len(train_dataset))
        dev_dataset = train_dataset[idx:]
        train_dataset = train_dataset[:idx]

        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True, collate_fn= utils.collate_fn)
        dev_dataloader = DataLoader(dev_dataset, self.test_batch, shuffle= True, collate_fn= utils.collate_fn)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = MNISTDataset(self.test_img_path, self.test_label_path)
        test_dataloader = DataLoader(test_dataset, self.train_batch, shuffle= False, collate_fn= utils.collate_fn)
        return test_dataloader