from models.convolucionais.vae_conv import ConvVAE
from experiments.dataloader import get_dataloaders
from experiments.train import train_vae
from experiments.utils import plot_reconstructions_vae, save_metrics

import torch
import os

device = ("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloaders(batch_size=128)

latent_dims = [2, 4, 8, 16, 32, 64, 128]

for latent_dim in latent_dims:
    print(f"\nTreinando Conv VAE com espaço latente = {latent_dim}")
    model = ConvVAE(latent_dim=latent_dim).to(device)
    metrics = train_vae(model, train_loader, test_loader, device, epochs=5, lr=1e-3)

    plot_reconstructions_vae(model, test_loader, device,
                             save_path=f"results/reconstructions/conv_vae/conv_vae_latent{latent_dim}.png")
    save_metrics(metrics, filename=f"results/metrics/conv_vae/conv_vae_latent{latent_dim}.csv")