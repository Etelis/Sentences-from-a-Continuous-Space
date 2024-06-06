import pytest
import torch
from models.lstm_encoder import LSTMEncoder

@pytest.fixture
def encoder():
    return LSTMEncoder(embedding_dim=10, hidden_dim=20, num_layers=2, bidirectional=False)

def test_lstm_encoder_forward(encoder):
    input_seq = torch.randn(5, 7, 10)  # batch_size=5, seq_len=7, embedding_dim=10
    seq_lengths = torch.tensor([7, 6, 5, 4, 3])
    
    output, (h_n, c_n) = encoder(input_seq, seq_lengths)
    
    assert output.shape == (5, 7, 20)  # (batch_size, seq_len, hidden_dim)
    assert h_n.shape == (2, 5, 20)  # (num_layers, batch_size, hidden_dim)
    assert c_n.shape == (2, 5, 20)  # (num_layers, batch_size, hidden_dim)
