import torch
import torch.nn as nn
from models.convolucionais.ae_conv import ConvAutoencoder

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvVAE, self).__init__()
        self.encoder = ConvAutoencoder(latent_dim).encoder
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = ConvAutoencoder(latent_dim).decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        h_flat = self.flatten(h)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        h_dec = self.fc_dec(z).view(-1, 256, 4, 4)
        x_hat = self.decoder(h_dec)
        return x_hat, mu, logvar
