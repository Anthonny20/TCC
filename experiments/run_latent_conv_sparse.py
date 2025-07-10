from models.convolucionais.sparse_conv import SparseConvAutoencoder
from experiments.dataloader import get_dataloaders
from experiments.train import train_autoencoder
from experiments.utils import plot_reconstructions_sparse, save_metrics
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Garante que as pastas de saída existem
os.makedirs("results/reconstructions/conv_sparse", exist_ok=True)
os.makedirs("results/metrics/conv_sparse", exist_ok=True)

# Define os tamanhos do espaço latente que serão testados
latent_dims = [2, 4, 8, 16, 32, 64, 128]

train_loader, test_loader = get_dataloaders(batch_size=128)

# Treinamento para cada valor de espaço latente
for latent_dim in latent_dims:
    print(f"\nTreinando Sparse AE com espaço latente = {latent_dim}")
    model = SparseConvAutoencoder(latent_dim=latent_dim).to(device)
    metrics = train_autoencoder(model, train_loader, test_loader, device, epochs=5, lr=1e-3)

    # Salva a reconstrução e métricas
    plot_reconstructions_sparse(model, test_loader, device,
                         save_path=f"results/reconstructions/conv_sparse/sparse_ae_latent{latent_dim}.png")
    save_metrics(metrics, filename=f"results/metrics/conv_sparse/sparse_ae_latent{latent_dim}.csv")