import torch
import torch.nn.functional as F
from piq import ssim, psnr
from sewar.full_ref import uqi, ergas
from math import sqrt
import numpy as np
from tqdm import tqdm
import os
import csv

def tensor_to_image(tensor):
    """
    Converte um tensor (B, C, H, W) ou (C, H, W) para numpy (H, W, C), float 32 [0, 255]
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return img


@torch.inference_mode()
def batch_metrics(x, y):
    # Garante que os tensores estejam no mesmo dispositivo e formato
    if isinstance(x, torch.Tensor):
        x = x.clamp(0, 1)
    if isinstance(y, torch.Tensor):
        y = y.clamp(0, 1)

    # PIQ (funciona por batch)
    mse_val = F.mse_loss(y, x).item()
    rmse_val = sqrt(mse_val)
    psnr_val = psnr(y, x, data_range=1.0, reduction='mean').item()
    ssim_val = ssim(y, x, data_range=1.0, reduction='mean').item()
    
    # SEWAR (funciona por imagem)
    x_img = tensor_to_image(x)
    y_img = tensor_to_image(y)
    uqi_val = uqi(x_img, y_img)
    ergas_val = ergas(x_img, y_img)

    return {
        'MSE': round(mse_val, 6),
        'RMSE': round(rmse_val, 6),
        'PSNR': round(psnr_val, 6),
        'SSIM': round(ssim_val, 6),
        'UQI': round(uqi_val, 6),
        'ERGAS': round(ergas_val, 6)
    }

@torch.inference_mode
def evaluate_loader(model, dataloader, device):
    model.eval()
    all_metrics = {k: [] for k in ['MSE', 'RMSE', 'PSNR', 'SSIM', 'UQI', 'ERGAS']}

    for images, _ in tqdm(dataloader, desc='Avaliando'):
        images = images.to(device)
        outputs = model(images)

        # Garante que as formas estão corretas
        if isinstance(outputs, tuple):  # se for VAE ou similar
            outputs = outputs[0]

        # Calcula métricas para o batch
        metrics = batch_metrics(images, outputs)

        # Salva métricas do batch
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    # Média de cada métrica
    averaged = {k: round(sum(v) / len(v), 6) for  k, v in all_metrics.items()}
    return averaged

def save_metrics(metrics, save_path, latent_dim):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    file_exists = os.path.isfile(save_path)

    fieldnames = ["latent_dim"] + list(metrics.keys())

    with open(save_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Escreve cabeçalho se for a primeira fez
        if not file_exists:
            writer.writeheader
        
        # Escreve linha de métricas
        row = {"latent_dim": latent_dim}
        row.update(metrics)
        writer.writerow(row)