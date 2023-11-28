from torch.utils.data import Dataset
import idx2numpy

class MNISTDataset(Dataset):
    def __init__(self, img_path, label_path):
        super().__init__()

        imgs = idx2numpy.convert_from_file(img_path)
        labels = idx2numpy.convert_from_file(label_path)

        self.__data = {}

        for idx, (img, label) in enumerate(zip(imgs, labels)):
            self.__data[idx] = {
                "image": img,
                "label": label
            }

    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            selected_data = [{"image": self.__data[i]["image"], "label": self.__data[i]["label"]} for i in range(start, stop, step)]
            return selected_data
        else:
            return self.__data[idx]
