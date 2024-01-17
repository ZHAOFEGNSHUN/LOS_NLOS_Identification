import numpy as np
import thop
import torch
import os
from torch.utils.data import Dataset, random_split, DataLoader


class CIRDataset(Dataset):
    def __init__(self, image_array, label_array):
        self.labels = torch.tensor(label_array, dtype=torch.float32)
        self.data = torch.tensor(image_array, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def generate_datasets(image_array_path, label_array_path):
    image_array = np.load(image_array_path)
    label_array = np.load(label_array_path)
    dataset = CIRDataset(image_array=image_array, label_array=label_array)
    # Split the dataset
    train_size_total = int(0.8 * len(dataset))
    train_size = int(0.8 * train_size_total)
    val_size = train_size_total - train_size
    test_size = len(dataset) - train_size_total
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def generate_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def compute_FLOPS_PARAMS(x, model):
    flops, params = thop.profile(model, inputs=(x,), verbose=False)
    print("FLOPs={:.2f}M".format(flops / 1e6))
    print("params={:.2f}M".format(params / 1e6))


def label_to_onehot(target, num_classes=2):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


if __name__ == '__main__':
    # Generate the datasets from the npy file
    label_array_path = '/Users/bytedance/Desktop/ZFS/LOS_NLOS_Identification/data/dataset/labels.npy'
    image_array_path = '/Users/bytedance/Desktop/ZFS/LOS_NLOS_Identification/data/grayscale_images/gray_images_part.npy'
    train_dataset, val_dataset, test_dataset = generate_datasets(image_array_path=image_array_path,
                                                                 label_array_path=label_array_path)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    # Generate the data loaders from datasets
    batch_size = 64
    train_loader, val_loader, test_loader = generate_loaders(train_dataset, val_dataset, test_dataset, batch_size)
