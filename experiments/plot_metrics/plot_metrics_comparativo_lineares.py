import os
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "results/metrics"

# Dicionário para armazenar os dados por modelo
model_dirs = {
    "AE Linear": "ae_linear",
    "AE Sparse": "ae_sparse",
    "AE Denoising": "ae_denoising",
    "VAE": "ae_vae"
}
# Dicionário para armazenar os resultados
results = {model: [] for model in model_dirs}

# Ler os resultados de cada modelo
for model_name, subdir in model_dirs.items():
    path = os.path.join(base_dir, subdir)
    for file in sorted(os.listdir(path)):
        if file.endswith(".csv") and "latent" in file:
            try:
                latent_dim = int(file.split("latent")[-1].split(".")[0])
                df = pd.read_csv(os.path.join(path, file))
                df.columns = df.columns.str.upper()
                mse = df['MSE'].values[0]
                ssim = df['SSIM'].values[0]
                results[model_name].append((latent_dim, mse, ssim))
            except Exception as e:
                print(f"Erro ao processar {file} ({model_name}): {e}")

# Plotar comparação MSE
plt.figure(figsize=(10, 5))
for model_name, data in results.items():
    if data:
        data.sort()
        latent_dims, mses, _ = zip(*data)
        plt.plot(latent_dims, mses, marker='o', label=model_name)
plt.xlabel("Dimensão do Espaço Latente")
plt.ylabel("MSE")
plt.title("Comparação de MSE entre os Modelos Lineares")
plt.legend()
plt.grid(True)
plt.savefig("results/metrics/mse_comparativo_lineares.png")
plt.show()

# Plotar comparação SSIM
plt.figure(figsize=(10, 5))
for model_name, data in results.items():
    if data:
        data.sort()
        latent_dims, _, ssims = zip(*data)
        plt.plot(latent_dims, ssims, marker='o', label=model_name)
plt.xlabel("Dimensão do Espaço Latente")
plt.ylabel("SSIM")
plt.title("Comparação de SSIM entre Modelos Lineares")
plt.legend()
plt.grid(True)
plt.savefig("results/metrics/ssim_comparativo_lineares.png")
plt.show()