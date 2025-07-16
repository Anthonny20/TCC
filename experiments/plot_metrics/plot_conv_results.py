import pandas as pd
import matplotlib.pyplot as plt
import os

models = ["conv_ae", "conv_sparse", "conv_denoising", "conv_vae"]
metrics = ["MSE", "RMSE", "PSNR", "SSIM", "UQI", "ERGAS"]

plt.style.use("seaborn-v0_8-colorbilnd")


for metric in metrics:
    plt.figure(figsize=(8, 5))
    for model in models:
        path = f"results/metrics/{model}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        plt.plot(df["latent_dim"], df[metric], label=model.replace("conv_","").upper(), marker="o")

    plt.xlabel("Tamanho do espaço latente")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Latent Dim (Modelos Convolucionais)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/plots/{metric.lower()}_conv.png")
    plt.close()