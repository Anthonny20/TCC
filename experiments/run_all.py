import matplotlib.pyplot as plt
import pandas as pd
from models.convolucionais.ae_conv import ConvAutoencoder
from models.convolucionais.sparse_conv import SparseConvAutoencoder
from models.convolucionais.denoising_conv import DenoisingConvAutoencoder
from models.convolucionais.vae_conv import ConvVAE
from experiments.train import *

models = {
    "conv_ae": ConvAutoencoder,
    "conv_sparse": SparseConvAutoencoder,
    "conv_denoising": DenoisingConvAutoencoder,
    "conv_vae": ConvVAE
}

latent_dims = [2, 4, 8, 16, 32, 64, 128, 256]

results = []

for name, ModelClass in models.items():
    for latent_dim in latent_dims:
        print(f"Treinando {name} com latent_dim={latent_dim}...")
        model = ModelClass(latent_dim=latent_dim).to(device)
        
        # Aqui chama a função de treino respectiva
        if name == "conv_ae":
            metrics = train_conv_autoencoder(model, train_loader, test_loader, device, epochs=30)
        elif name == "conv_sparse":
            metrics = train_conv_sparse_autoenconder(model, train_loader, test_loader, device, epochs=30)
        elif name == "conv_denoising":
            metrics = train_conv_denoising_autoencoder(model, train_loader, test_loader, device, epochs=30)
        elif name == "conv_vae":
            metrics = train_conv_vae(model, train_loader, test_loader, device, epochs=30)
        
        metrics["model"] = name
        metrics["latent_dim"] = latent_dim
        
        # Salva no csv, função save_metrics pode ser chamada aqui ou você pode salvar depois tudo de uma vez
        save_metrics(metrics, save_path=f"results/metrics/{name}.csv", latent_dim=latent_dim)
        
        results.append(metrics)

# Após tudo, converte para DataFrame para plotar gráficos comparativos
df_results = pd.DataFrame(results)

# Exemplo: plotar MSE para todos modelos e latent dims
plt.figure(figsize=(10,6))
for name in models.keys():
    subset = df_results[df_results['model'] == name]
    plt.plot(subset['latent_dim'], subset['MSE'], label=name)
plt.xlabel("Latent Dimension")
plt.ylabel("MSE")
plt.title("MSE vs Latent Dimension por Modelo")
plt.legend()
plt.grid(True)
plt.savefig("results/plots/mse_comparacao.png")
plt.show()
