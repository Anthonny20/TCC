import torch
from models.linear.ae import Autoencoder

# Denoising Autoencoder Linear
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, latent_dim=32):
        super(DenoisingAutoencoder, self).__init__(latent_dim)

    def forward(self, x):
        noise = torch.randn_like(x) * 0.2
        x_noisy = x + noise
        x_noisy = torch.clamp(x_noisy, 0., 1.)
        z = self.encoder(x_noisy)
        return self.decoder(z), z