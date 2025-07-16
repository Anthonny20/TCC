import torch
import torch.nn as nn

# Autoencoder Básico Convolucional
class ConvAutoencoder(nn.Module):

    """
    Por que essa arquitetura é adequada para CIFAR‑10?
        Mais canais iniciais (3 → 32)
        Necessário para captar informação de cor e bordas finas; 32 filtros é um ponto de partida clássico.

        Três blocos de downsampling (32 → 16 → 8 → 4 px)
        • Mantém a profundidade moderada (≈ 10 – 15 layers efetivas).
        • 4 × 4 de resolução + 128 filtros gera uma representação compacta, porém expressiva.
        • Reduz drasticamente o tamanho antes do Flatten, evitando vetor gigantesco.

        Latent vector (latent_dim) separado
        Facilita experimentar latent_dim ∈ {32, 64, 128, 256} sem refatorar o CNN.

        Decoder espelhado
        ConvTranspose assegura que cada upsample reflita o downsample correspondente, gerando saídas 32×32×3.

        Sigmoid() final
        Mantém pixels normalizados em [0, 1] (combina com transforms.ToTensor()).


    """
    def __init__(self, latent_dim=32):
        super(ConvAutoencoder, self).__init__()
        # Codificador
        self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 32, 32]
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 16, 16]
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [B, 128, 8, 8]
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # [B, 256, 4, 4]
        nn.BatchNorm2d(256),
        nn.ReLU()
    )
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(256 * 4 * 4, latent_dim)
        
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),  # [B, 256, 8, 8]
        nn.Conv2d(256, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Upsample(scale_factor=2),  # [B, 128, 16, 16]
        nn.Conv2d(128, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.Upsample(scale_factor=2),  # [B, 64, 32, 32]
        nn.Conv2d(64, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 3, 3, padding=1),
        nn.Sigmoid()
    )
        


        

    def forward(self, x):
        h = self.encoder(x)
        z = self.fc_enc(self.flatten(h))       # vetor latente
        h_dec = self.fc_dec(z).view(-1, 256,4,4)
        x_hat = self.decoder(h_dec)
        return x_hat