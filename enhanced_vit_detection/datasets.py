import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, train=True):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transforms.ToTensor())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]