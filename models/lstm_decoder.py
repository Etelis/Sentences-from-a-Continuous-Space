import torch
from torch import nn

class LSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_to_vocab = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_seq, hidden):
        decoder_output, hidden = self.lstm(input_seq, hidden)
        logits = self.output_to_vocab(decoder_output)
        return logits, hidden
