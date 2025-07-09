import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch.nn.functional as F

def plot_reconstructions(model, dataloader, device, save_path):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.to(device)
    with torch.no_grad():
        outputs, _ = model(images)  # Corrigido aqui

    fig, axs = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        axs[0, i].imshow(images[i, 0].cpu(), cmap='gray')  # pegar canal 0 direto
        axs[1, i].imshow(outputs[i, 0].cpu(), cmap='gray') # pegar canal 0 direto
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics(metrics, filename):
    df = pd.DataFrame([metrics])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

def compute_mse(original, reconstructed):
    if isinstance(reconstructed, tuple):
        reconstructed = reconstructed[0]  # usa apenas a reconstrução
    return F.mse_loss(reconstructed, original).item()


def compute_ssim(original, reconstructed):
    # Se reconstructed for tupla (ex: VAE), pega só a imagem reconstruída
    if isinstance(reconstructed, tuple):
        reconstructed = reconstructed[0]

    # Remove o batch (deixa [C, H, W] ou [H, W])
    original = original.detach().cpu().squeeze().numpy()
    reconstructed = reconstructed.detach().cpu().squeeze().numpy()

    # Se tiver canal (1, H, W), remove o canal para SSIM 2D
    if original.ndim == 3 and original.shape[0] == 1:
        original = original[0]
    if reconstructed.ndim == 3 and reconstructed.shape[0] == 1:
        reconstructed = reconstructed[0]

    # Confere se shapes são iguais
    if original.shape != reconstructed.shape:
        raise ValueError(f"SSIM: As imagens não têm o mesmo shape. Original: {original.shape}, Reconstruída: {reconstructed.shape}")

    return ssim(original, reconstructed, data_range=1.0)
