import torch
from torch import nn

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
    
    def forward(self, input_seq, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(input_seq, seq_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden
