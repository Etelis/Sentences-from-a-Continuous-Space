import pytest
import torch
from models.lstm_decoder import LSTMDecoder

@pytest.fixture
def decoder():
    return LSTMDecoder(embedding_dim=10, hidden_dim=20, num_layers=2, vocab_size=100)

def test_lstm_decoder_forward(decoder):
    input_seq = torch.randn(5, 7, 10)  # batch_size=5, seq_len=7, embedding_dim=10
    h0 = torch.randn(2, 5, 20)  # num_layers=2, batch_size=5, hidden_dim=20
    c0 = torch.randn(2, 5, 20)  # num_layers=2, batch_size=5, hidden_dim=20

    hidden = (h0, c0)

    logits, hidden = decoder(input_seq, hidden)
    
    assert logits.shape == (5, 7, 100)  # (batch_size, seq_len, vocab_size)
    assert hidden[0].shape == (2, 5, 20)  # (num_layers, batch_size, hidden_dim)
    assert hidden[1].shape == (2, 5, 20)  # (num_layers, batch_size, hidden_dim)
