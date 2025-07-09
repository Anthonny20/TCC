from models.linear.denoising import DenoisingAutoencoder
from experiments.dataloader import get_dataloaders
from experiments.train import train_autoencoder
from experiments.utils import plot_reconstructions, save_metrics

import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloaders(batch_size=128)

latent_dims = [2, 4, 8, 16, 32, 64, 128]

for latent_dim in latent_dims:
    print(f"\nTreinando Denoising AE com espaço latente = {latent_dim}")
    model = DenoisingAutoencoder(latent_dim=latent_dim).to(device)
    metrics = train_autoencoder(model, train_loader, test_loader, device, epochs=5, lr=1e-3)

    # Salvar reconstrução e métricas
    plot_reconstructions(model, test_loader, device,
                         save_path=f"results/reconstructions/ae_denoising_latent{latent_dim}.png")
    save_metrics(metrics, filename=f"results/metrics/ae_denoising_latent{latent_dim}.csv")