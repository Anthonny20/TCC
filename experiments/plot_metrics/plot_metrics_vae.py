import pandas as pd
import os
import matplotlib.pyplot as plt

metrics_dir = "results/metrics/ae_vae"
results = []

for file in sorted(os.listdir(metrics_dir)):
    if file.startswith("ae_vae_latent") and file.endswith(".csv"):
        latent_dim = int(file.split("latent")[-1].split(".")[0])
        df = pd.read_csv(os.path.join(metrics_dir, file))
        df.columns = df.columns.str.upper()
        try:
            mse = df['MSE'].values[0]
            ssim = df['SSIM'].values[0]
            results.append((latent_dim, mse, ssim))
        except Exception as e:
            print(f"Erro ao processar {file}: {e}")

results.sort(key=lambda x: x[0])
latent_dims, mses, ssims  = zip(*results)

plt.figure(figsize=(10, 5))
plt.plot(latent_dims, mses, marker='o', color="red")
plt.xlabel("Dimensão do Espaço Latente")
plt.ylabel("Erro Quadrático Médio (MSE)")
plt.title("MSE vs. Latent Dim - VAE Linear")
plt.grid(True)
plt.savefig("results/logs/ae_vae/mse_plot_vae_linear.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(latent_dims, ssims, marker='o', color="blue")
plt.xlabel("Dimensão do Espaço Latente")
plt.ylabel("Índice de Similaridade Estrutural (SSIM)")
plt.title("SSIM vs. Latent Dim - VAE Linear")
plt.grid(True)
plt.savefig("results/logs/ae_vae/ssim_plot_vae_linear.png")
plt.show()