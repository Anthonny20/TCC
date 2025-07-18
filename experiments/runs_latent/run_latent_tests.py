from models.convolucionais.ae_conv import ConvAutoencoder
from models.convolucionais.sparse_conv import SparseConvAutoencoder
from models.convolucionais.denoising_conv import DenoisingConvAutoencoder
from models.convolucionais.vae_conv import ConvVAE
from experiments.dataloader import get_dataloaders
from experiments.train import *
from experiments.reconstructions import *
from experiments.run_all import save_metrics
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_dataloaders(batch_size=128, dataset="CIFAR10")

# Vários valores para testar
latent_dims = [2, 4, 8, 16, 32, 64, 128, 256]

models_info = {
    "conv_ae": {
        "model_class": ConvAutoencoder,
        "train_fn": train_conv_autoencoder,
    },
    "conv_sparse": {
        "model_class": SparseConvAutoencoder,
        "train_fn": train_conv_sparse_autoenconder,
        "extra_args": {"sparsity_weight": 1e-4}
    },
    "conv_denoising": {
        "model_class": DenoisingConvAutoencoder,
        "train_fn": train_conv_denoising_autoencoder,
        "extra_args": {"noise_factor": 0.3}
    },
    "conv_vae": {
        "model_class": ConvVAE,
        "train_fn": train_conv_vae,
    }
}

for latent_dim in latent_dims:
    print(f"\nTreinando Autoencoder com espaço latente = {latent_dim}")
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)  # já manda pra device

    # Teste rápido para checar shapes antes do treinamento
    images, _ = next(iter(train_loader))
    images = images.to(device)
    print('Input images shape:', images.shape)

    outputs, z = model(images)
    print('Output shape:', outputs.shape)

    # Treinamento
    metrics = train_conv_autoencoder(
        model, 
        train_loader, 
        test_loader, 
        device, 
        epochs=30, 
        lr=1e-3
        )

    # Diretórios
    recon_path = os.path.join("results", "reconstructions", f"ae_linear_latent{latent_dim}.png")
    metric_path = os.path.join("results", "metrics", f"ae_linear_latent{latent_dim}.csv")

    # Salvar saída
    save_reconstructions(model, test_loader, device, save_path=recon_path)
    save_metrics(metrics, filename=metric_path)

    print(f"  > MSE: {metrics['mse']:.4f} | RMSE {metrics['rsme']:.4f} | PSNR {metrics['psnr']:.4f} | SSIM {metrics['ssim']:.4f} | UQI {metrics['uqi']:.4f} | ERGAS {metrics['ergas']:.4f}")
