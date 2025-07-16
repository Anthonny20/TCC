import torch
import torch.nn as nn
from models.convolucionais.ae_conv import ConvAutoencoder

class DenoisingConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(DenoisingConvAutoencoder, self).__init__()
        self.encoder = ConvAutoencoder(latent_dim).encoder
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = ConvAutoencoder(latent_dim).decoder

    def forward(self, x):
        h = self.encoder(x)
        z = self.fc_enc(self.flatten(h))
        h_dec = self.fc_dec(z).view(-1, 256, 4, 4)
        x_hat = self.decoder(h_dec)
        return x_hat