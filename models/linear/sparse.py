import torch
from ae import Autoencoder

# Sparse Autoencoder Linear (L1 penalty)
class SparseAutoencoder(Autoencoder):
    def __init__(self, latent_dim=32, sparsity_weight=1e-3):
        super(SparseAutoencoder, self).__init__(latent_dim)
        self.sparsity_weight = sparsity_weight
    
    def sparsity_loss(self, z):
        return self.sparsity_weight * torch.mean(torch.abs(z))