import torch

def collate_fn(Dataset):
    imgs = [data["image"].unsqueeze(0) for data in Dataset]
    labels = [data["label"].unsqueeze(0) for data in Dataset]

    return [torch.cat(imgs).type(torch.float32), torch.cat(labels).type(torch.LongTensor)]