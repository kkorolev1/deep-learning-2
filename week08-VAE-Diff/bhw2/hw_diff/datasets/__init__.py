import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image

# TODO: Remove later
from torchvision.datasets import MNIST

class ImagesDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        
        self.paths = list(Path(root).iterdir())
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = self.transform(image)
        return (image,)
    
    def __len__(self):
        return len(self.paths)


class MNISTWrapper(Dataset):
    def __init__(self, root, is_train):
        super().__init__()
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = MNIST(root, is_train, transform, download=True)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


__all__ = [
]