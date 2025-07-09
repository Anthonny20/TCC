import pandas as pd
import matplotlib.pyplot as plt
import os

# Caminho onde os arquicos CSV estão salvos
metrics_dir = "results/metrics"

# List para armazenar os resultados
results = []

# Lê cada arquivo CVS de métrica
for file in sorted(os.listdir(metrics_dir)):
    if file.endswith(".csv"):
        latent_dim = int(file.split("latent")[-1].split(".")[0])
        df = pd.read_csv(os.path.join(metrics_dir, file))
        df.columns = df.columns.str.upper()
        try:
            mse = df["MSE"].values[0]
            ssim = df["SSIM"].values[0]
            results.append((latent_dim, mse, ssim))
        except Exception as e:
            print(f"Erro ao processar {file}: {e}")
# Ordenar por latent_dim
results.sort(key=lambda x: x[0])

# Separar listas
latent_dims, mses, ssims = zip(*results)

# Plotar MSE
plt.figure(figsize=(10, 5))
plt.plot(latent_dims, mses, marker='o', label="MSE", color="red")
plt.xlabel("Dimensão do Espaço Latente")
plt.ylabel("Erro Quadrático Médio (MSE)")
plt.title("MSE vs. Latent Dim - Autoenconder Linear")
plt.grid(True)
plt.savefig("results/metrics/mse_plot_ae_linear.png")
plt.show()

# Plotar SSIM
plt.figure(figsize=(10, 5))
plt.plot(latent_dims, ssims, marker='o', label="SSIM", color="blue")
plt.xlabel("Dimensão do Espaço Latente")
plt.ylabel("Índice de Similaridade Estrutural (SSIM)")
plt.title("SSIM vs. Latent Dim - Autoenconder Linear")
plt.grid(True)
plt.savefig("results/metrics/ssim_plot_ae_linear.png")
plt.show()