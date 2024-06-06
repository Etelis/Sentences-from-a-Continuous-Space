import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, embedded_inputs, seq_lengths):
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        embedded_inputs = embedded_inputs[perm_idx]

        assert embedded_inputs.size(0) == seq_lengths.size(0), "Mismatch between embedded inputs and sequence lengths"
        packed_input = rnn_utils.pack_padded_sequence(embedded_inputs, seq_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, hidden = self.lstm(packed_input)

        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        _, unperm_idx = perm_idx.sort(0)
        output = output[unperm_idx]
        if isinstance(hidden, tuple):  # LSTM case
            hidden = (hidden[0][:, unperm_idx, :], hidden[1][:, unperm_idx, :])
        else:
            hidden = hidden[:, unperm_idx, :]

        return output, hidden
