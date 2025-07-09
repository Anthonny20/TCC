import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=128, dataset='FashionMNIST'):
    transform = transforms.ToTensor()

    if dataset == 'MNIST':
        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    elif dataset == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Dataset não suportado. Use 'MNIST' ou 'FashionMNIST'")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader    