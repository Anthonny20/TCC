import torch
from models.convolucionais.ae_conv import ConvAutoencoder
from models.convolucionais.sparse_conv import SparseConvAutoencoder
from models.convolucionais.denoising_conv import DenoisingConvAutoencoder
from models.convolucionais.vae_conv import ConvVAE
from experiments.dataloader import get_dataloaders
from experiments.evaluate import save_metrics, save_reconstructions
from experiments.train import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_dataloaders(batch_size=128, dataset="CIFAR10")

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

for model_name, info in models_info.items():
    for latent_dim in latent_dims:
        print(f"\nTreinando {model_name} com latent_dim = {latent_dim}")
        model = info["model_class"](latent_dim=latent_dim).to(device)

        train_fn = info["train_fn"]
        extra_args = info.get("extra_args", {})

        # Chama função de treino passando latent_dim para exibir no print
        metrics = train_fn(
            model,
            train_loader,
            test_loader,
            device,
            epochs=30,
            lr=1e-3,
            latent_dim=latent_dim,
            **extra_args
        )

        # Salvar resultados em csv específico para cada modelo
        save_metrics(metrics,
                     save_path=f"results/metrics/{model_name}.csv",
                     latent_dim=latent_dim)
        
        #Salvar reconstruções
        model.eval()
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                save_reconstructions(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    save_dir=f"results/reconstructions/{model_name}/latent_{latent_dim}"
                )
                break #
