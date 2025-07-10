import torch
import torch.nn as nn
import torch.nn.functional as F

# Sparse Autoencoder Convolutional
class SparseConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32, sparsity_weight=1e-5):
        super(SparseConvAutoencoder, self).__init__()
        self.sparsity_weight = sparsity_weight

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(32 * 7 * 7, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 32 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        flat = encoded.view(x.size(0),  -1)
        latent = self.fc1(flat)

        # Penalização L1 (sparse loss) fica a cargo da função de treino
        decoded = self.fc2(latent)
        decoded = decoded.view(x.size(0), 32, 7, 7)
        decoded = self.decoder(decoded)
        return decoded, latent
