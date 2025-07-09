from models.linear.ae import Autoencoder
from models.linear.sparse import SparseAutoencoder
from models.linear.denoising import DenoisingAutoencoder
from models.linear.vae import VariaotinalAutoencoder
from experiments.dataloader import get_dataloaders
from experiments.train import train_autoencoder
from experiments.utils import plot_reconstructions, save_metrics
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloaders(batch_size=128)

# Vários valores para testar
latent_dims = [2, 4, 8, 16, 32, 64, 128]

for latent_dim in latent_dims:
    print(f"\nTreinando Autoencoder com espaço latente = {latent_dim}")
    model = Autoencoder(latent_dim=latent_dim).to(device)  # já manda pra device

    # Teste rápido para checar shapes antes do treinamento
    images, _ = next(iter(train_loader))
    images = images.to(device)
    print('Input images shape:', images.shape)

    outputs, z = model(images)
    print('Output shape:', outputs.shape)

    # Treinamento
    metrics = train_autoencoder(
        model, 
        train_loader, 
        test_loader, 
        device, 
        epochs=5, 
        lr=1e-3
        )

    # Diretórios
    recon_path = os.path.join("results", "reconstructions", f"ae_linear_latent{latent_dim}.png")
    metric_path = os.path.join("results", "metrics", f"ae_linear_latent{latent_dim}.csv")

    # Salvar saída
    plot_reconstructions(model, test_loader, device, save_path=recon_path)
    save_metrics(metrics, filename=metric_path)

    print(f"  > MSE: {metrics['mse']:.4f} | SSIM {metrics['ssim']:.4f}")
