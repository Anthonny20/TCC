from models.linear.vae import VariaotinalAutoencoder
from experiments.dataloader import get_dataloaders
from experiments.train import train_vae
from experiments.utils import plot_reconstructions, save_metrics, plot_reconstructions_vae


import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloaders(batch_size=128)

latent_dims = [2, 4, 8, 16, 32, 64, 128]

for latent_dim in latent_dims:
    print(f"\nTreinando VAE com espaço latente = {latent_dim}")
    model = VariaotinalAutoencoder(latent_dim=latent_dim).to(device)

    metrics = train_vae(model, train_loader, test_loader, device, epochs=5, lr=1e-3)

    # Salvar reconstrução e métricas
    plot_reconstructions_vae(model, test_loader, device,
                         save_path=f"results/reconstructions/ae_vae_latent{latent_dim}.png")
    save_metrics(metrics, filename=f"results/metrics/ae_vae_latent{latent_dim}.csv")