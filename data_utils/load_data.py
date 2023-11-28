from data_utils import MNISTDataset, utils
from torch.utils.data import DataLoader

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
        train_dataset = MNISTDataset.MNISTDataset(self.train_img_path, self.train_label_path)
        dev_dataset = train_dataset[0.8*len(train_dataset):]
        train_dataset = train_dataset[:0.8*len(train_dataset)]

        train_dataloader = DataLoader(train_dataset, self.train_batch, shuffle= True, collate_fn= utils.collate_fn)
        dev_dataloader = DataLoader(dev_dataset, self.test_batch, shuffle= True, collate_fn= utils.collate_fn)

        return train_dataloader, dev_dataloader
    
    def load_test(self):
        test_dataset = MNISTDataset.MNISTDataset(self.test_img_path, self.test_label_path)
        test_dataloader = DataLoader(test_dataset, self.train_batch, shuffle= False, collate_fn= utils.collate_fn)
        return test_dataloader