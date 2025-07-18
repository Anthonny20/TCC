import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from experiments.evaluate import batch_metrics

# AUTOENCODER CONV
def train_conv_autoencoder(model, train_loader, test_loader, device, epochs=30, lr=1e-3, latent_dim=None, patience=5, min_delta=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        losses = []
        loop = tqdm(train_loader, desc=f"ConvAE [Epoch {epoch+1}/{epochs}]")
        for images, _ in loop:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f"epoca {epoch+1}: latent_dim: {latent_dim} loss: {avg_loss:.4f}")
        
        
        # Early stopping check
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Parando o treino na época {epoch+1} por falta de melhora na loss.")
            break

    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        metrics = batch_metrics(images, outputs)
        return metrics

    
# SPARSE AUTOENCODER CONV
def train_conv_sparse_autoenconder(model, train_loader, test_loader, device, epochs=30, lr=1e-3, sparsity_weight=1e-4, latent_dim=None, patience=5, min_delta=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        losses = []
        loop = tqdm(train_loader, desc=f"SparseConvAE [Epoch {epoch+1}/{epochs}]")
        for images, _ in loop:
            images = images.to(device)
            outputs, latent = model(images)
            loss_recon = criterion(outputs, images)
            loss_sparse = torch.mean(torch.abs(latent))
            loss = loss_recon + sparsity_weight * loss_sparse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f"epoca {epoch+1}: latent_dim: {latent_dim} loss: {avg_loss:.4f}")
        
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping na época {epoch+1} - sem melhora na loss.")
            break

    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        outputs, _ = model(images)
        metrics = batch_metrics(images, outputs)
        return metrics

    

# DENOISING AUTOENCODER CONV
def train_conv_denoising_autoencoder(model, train_loader, test_loader, device, epochs=30, lr=1e-3, noise_factor=0.3, latent_dim=None, patience=5, min_delta=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        losses = []
        loop = tqdm(train_loader, desc=f"DenoisingConvAE [Epoch {epoch+1}/{epochs}]")
        for images, _ in loop:
            images = images.to(device)
            noisy = images + noise_factor * torch.randn_like(images)
            noisy = torch.clamp(noisy, 0., 1.)
            outputs = model(noisy)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f"epoca {epoch+1}: latent_dim: {latent_dim} loss: {avg_loss:.4f}")
        
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping na época {epoch+1} - sem melhora na loss.")
            break

    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        noisy = images + noise_factor * torch.randn_like(images)
        noisy = torch.clamp(noisy, 0., 1.)
        outputs = model(noisy)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        metrics = batch_metrics(images, outputs)
        return metrics

    
# VARIATIONAL AUTOENCODER CONV

def train_conv_vae(model, train_loader, test_loader, device, epochs=30, lr=1e-3, latent_dim=None, patience=5, min_delta=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    epochs_no_improve = 0

    model.train()
    for epoch in range(epochs):
        losses = []
        loop = tqdm(train_loader, desc=f"VAEConv [Epoch {epoch+1}/{epochs}]")
        for images, _ in loop:
            images = images.to(device)
            recon, mu, logvar = model(images)

            recon_loss = F.mse_loss(recon, images, reduction='sum') / images.size(0)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
            loss = recon_loss + kl_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f"epoca {epoch+1}: latent_dim: {latent_dim} loss: {avg_loss:.4f}")
        
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping na época {epoch+1} - sem melhora na loss.")
            break

    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        recon, _, _ = model(images)
        metrics = batch_metrics(images, recon)
        return metrics
