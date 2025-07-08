import torch
import torch.nn as nn
import torch.nn.functional as F

# Sparse Autoencoder Convolutional
class SparseConvAutoencoder(nn.Module):
    def __init__(self):
        super(SparseConvAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose(16, 1, 3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z
