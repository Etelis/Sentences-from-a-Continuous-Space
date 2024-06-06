import torch
from torch import nn

class LSTMDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, vocab_size):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_to_vocab = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, inp, hidden):
        dec_output, hidden = self.lstm(inp, hidden)
        logits = self.output_to_vocab(dec_output)
        return logits, hidden
