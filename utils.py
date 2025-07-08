import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Download Datasets
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root='C:\\Users\\anthonny.paz\\Documents\\GitHub\\TCC\\data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='C:\\Users\\anthonny.paz\\Documents\\GitHub\\TCC\\data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
