import torch
from torch import nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, hidden_size, latent_dim, num_layers, bidirectional=False):
        super(VariationalAutoencoder, self).__init__()
        self.hidden_size_factor = 2 if bidirectional else 1
        self.encoder_to_mu = nn.Linear(self.hidden_size_factor * num_layers * hidden_size, latent_dim)
        self.encoder_to_logsigma = nn.Linear(self.hidden_size_factor * num_layers * hidden_size, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, self.hidden_size_factor * num_layers * hidden_size)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, enc_hidden):
        mu = self.encoder_to_mu(enc_hidden)
        logsigma = self.encoder_to_logsigma(enc_hidden)
        sigma = torch.exp(0.5 * logsigma)
        z = mu + sigma * self.N.sample(mu.shape).to(mu.device)
        return z, mu, logsigma
