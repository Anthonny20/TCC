import torch
from torch import nn, optim
from tqdm import tqdm
from experiments.utils import compute_mse, compute_ssim

def train_autoencoder(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch[{epoch+1}/{epochs}]")
        for batch in loop:
            images, _ = batch
            images = images.to(device)
            outputs = model(images)
                        # Verifica se outputs é uma tupla (ex: VAE retorna (recon, mu, logvar), AE pode retornar (recon, latente))
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # pega apenas a reconstrução

            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    # Avaliação após treino
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        mse = compute_mse(images, outputs)
        ssim_value = compute_ssim(images[0], outputs[0][0])
    return {'mse':mse, 'ssim':ssim_value}