import torch
from torch.utils.data import Dataset
import idx2numpy

class MNISTDataset(Dataset):
    def __init__(self,transform, img_path, label_path):
        super().__init__()

        imgs = idx2numpy.convert_from_file(img_path)
        labels = idx2numpy.convert_from_file(label_path)
    
        self.__data = [{"image": transform(torch.tensor(img).type(torch.float32)) ,"label": torch.tensor(label)} for img, label in zip(imgs, labels)]

    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.start or 0, index.stop or len(self), index.step or 1
            selected_data = [{"image": self.__data[i]["image"], "label": self.__data[i]["label"]} for i in range(start, stop, step)]
            return selected_data
        else:
            return self.__data[index]
