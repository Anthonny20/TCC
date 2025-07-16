import os
import torch
from experiments.evaluate import evaluate_loader, save_metrics
from experiments.dataloader import get_dataloaders
from models.convolucionais.ae_conv import ConvAutoencoder
from models.convolucionais.sparse_conv import SparseConvAutoencoder
from models.convolucionais.denoising_conv import DenoisingConvAutoencoder
from models.convolucionais.vae_conv import ConvVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, test_loader = get_dataloaders(batch_size=128, dataset="CIFAR10")

# Diretório onde os modelos treinaodos estão salvos 

models = {
    "conv_ae": ConvAutoencoder,
    "conv_sparse": SparseConvAutoencoder,
    "conv_denoising": DenoisingConvAutoencoder,
    "conv_vae": ConvVAE
}

latent_dims = [2, 4, 8, 16, 32, 64, 128, 256]

for name, model_class in models.items():
    for latent_dim in latent_dims:
        print(f"\nAvaliando modelo {name} com o espaço latente = {latent_dim}")
        model = model_class(latent_dim=latent_dim).to(device)

        # Caminho para salvar os resultados
        result_path = f"results/metrics/{name}.csv"

        # Avaliação
        metrics = evaluate_loader(model, test_loader, device)

        # Salvando
        save_metrics(metrics, save_path=result_path, latent_dim=latent_dim)

