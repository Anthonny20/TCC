import torch
import torch.nn as nn

# Autoencoder Básico Convolucional
class ConvAutoencoder(nn.Module):

    """
    Explicação Rápida de como funciona:
        Parte                       Operação                        Tam Saída
        Conv2d(1, 16, 3, 2, 1)      reduz resoução para 14x14       [B, 16, 14, 14]
        Conv2d(16, 32, 3, 2, 1)     reduz para 7x7                  [B, 32, 7, 7]
        Flatten + Linear            gera vetor latente              [B, latent_dim]
        Linear + Unflatten          reconstrói para [32, 7, 7]     
        ConvTranspose2d             aumenta gradualmente até 28x28  [B, 1, 28, 28] 
    """
    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()
        # Codificador
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU(),
            nn.Flatten(),                                           # [B, 32*7*7]
            nn.Linear(32 * 7 * 7, latent_dim)                       # [B, latent_dim]
        )
        
        # Decodificador
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
