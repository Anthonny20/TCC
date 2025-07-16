import torch
from torch import nn, optim
import torch.nn.functional as F
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

def train_vae(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, _) in enumerate(tqdm(train_loader, desc=f"Epoch[{epoch+1}/{epochs}]")):
            images = images.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(images)
            recon_loss = F.mse_loss(recon, images)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
            loss = recon_loss + kl_div

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Avaliação
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)
    recon, _, _ = model(images)

    mse = compute_mse(images, recon)
    ssim = compute_ssim(images[0], recon[0])

    return {"MSE": mse, "SSIM": ssim}

def train_sparse_autoencoder(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Epoch[{epoch+1}/{epochs}]"):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images) + 1e-5 * torch.norm(outputs, 1) # Penalidade L1
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss:{running_loss/len(train_loader):.4f}")
    
    # Avaliar no teste
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):            # ← proteção
                outputs = outputs[0]
            break
    
    mse = compute_mse(images, outputs)
    ssim_value = compute_ssim(images[0], outputs[0])
    return {"MSE": mse, "SSIM": ssim_value}

def train_denoising_autoencoder(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Epoch[{epoch+1}/{epochs}]"):
            images = images.to(device)
            noisy_images = images + 0.3 * torch.randn_like(images) # Adiciona ruído
            noisy_images = torch.clamp(noisy_images, 0., 1.)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Avaliar no teste
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            noisy_images = images + 0.3 * torch.randn_like(images)
            noisy_images = torch.clamp(noisy_images, 0., 1.)
            outputs = model(noisy_images)
            if isinstance(outputs, tuple):            # ← proteção
                outputs = outputs[0]
            break

    mse = compute_mse(images, outputs)
    ssim_value = compute_ssim(images[0], outputs[0])
    return {"MSE": mse, "SSIM": ssim_value}

def train_con_vae(model, train_loader, test_loader, device, epochs=5, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in tqdm(train_loader, desc=f"Epocj[{epoch+1}/{epochs}]"):
            images = images.to(device)
            outputs, mu, logvar = model(images)

            # VAE loss = reconstrução + divergência KL
            recon_loss = F.mse_loss(outputs, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_loss) / images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
    
    # Avaliar
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs, _, _ = model(images)
            if isinstance(outputs, tuple):            # ← proteção
                outputs = outputs[0]
            break
    
    mse = compute_mse(images, outputs)
    ssim_value = compute_ssim(images[0], outputs[0])
    return {"MSE": mse, "SSIM": ssim_value}