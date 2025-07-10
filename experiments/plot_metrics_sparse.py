import pandas as pd
import matplotlib.pyplot as plt
import os

metrics_dir = "results/metrics/ae_sparse"
results = []

for file in sorted(os.listdir(metrics_dir)):
    if file.startswith("ae_sparse_latent") and file.endswith(".csv"):
        latent_dim = int(file.split("latent")[-1].split(".")[0])
        df = pd.read_csv(os.path.join(metrics_dir, file))
        df.columns = df.columns.str.upper()
        try:
            mse = df["MSE"].values[0]
            ssim = df["SSIM"].values[0]
            results.append((latent_dim, mse, ssim))
        except Exception as e:
            print(f"Erro ao processar {file}: {e}")

results.sort(key=lambda x: x[0])
latent_dims, mses, ssims = zip(*results)

plt.figure(figsize=(10, 5))
plt.plot(latent_dims, mses, marker='o', color='red')
plt.xlabel("Dimensão do Espaço Latente")
plt.ylabel("Erro Quadrático Médio (MSE)")
plt.title("MSE vs. Latent Dim - Sparse AE Linear")
plt.grid(True)
plt.savefig("results/logs/ae_sparse/mse_plot_sparse_ae_linear.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(latent_dims, ssims, marker='o', color='blue')
plt.xlabel("Dimensão do Espaço Latente")
plt.ylabel("Índice de Similiraridade Estrutural (SSIM)")
plt.title("SSIM vs. Latent Dim - Sparse AE Linear")
plt.grid(True)
plt.savefig("results/logs/ae_sparse/ssim_plot_sparse_ae_linear.png")
plt.show()