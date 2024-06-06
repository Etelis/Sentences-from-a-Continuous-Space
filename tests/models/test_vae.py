import pytest
import torch
from models.vae import VariationalAutoencoder

@pytest.fixture
def vae():
    return VariationalAutoencoder(hidden_dim=20, latent_dim=15, num_layers=2, bidirectional=False)

def test_vae_forward(vae):
    encoder_hidden = torch.randn(5, 40)  # batch_size=5, hidden_dim * num_layers
    
    z, mu, logsigma = vae(encoder_hidden)
    
    assert z.shape == (5, 15)  # (batch_size, latent_dim)
    assert mu.shape == (5, 15)  # (batch_size, latent_dim)
    assert logsigma.shape == (5, 15)  # (batch_size, latent_dim)
