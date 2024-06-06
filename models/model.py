import torch
from torch import nn
import torch.nn.functional as F
from .lstm_encoder import LSTMEncoder
from .lstm_decoder import LSTMDecoder
from .vae import VariationalAutoencoder

class SentenceFromSpaceModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, word_dropout_rate, embedding_dropout_rate, 
                 latent_dim, special_tokens, max_gen_len, bidirectional=False, num_layers=1):
        
        super(SentenceFromSpaceModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_dropout_rate = word_dropout_rate
        self.latent_dim = latent_dim
        self.special_tokens = special_tokens
        self.max_gen_len = max_gen_len
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = LSTMEncoder(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.vae = VariationalAutoencoder(hidden_dim, latent_dim, num_layers, bidirectional=bidirectional)
        self.decoder = LSTMDecoder(embedding_dim, hidden_dim, num_layers=num_layers, vocab_size=vocab_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_seq, seq_lengths):
        batch_size = input_seq.shape[0]

        embeddings = self.embedding_layer(input_seq)

        encoder_output, encoder_hidden = self.encoder(embeddings, seq_lengths)
        if isinstance(encoder_hidden, tuple):  # LSTM case
            encoder_hidden = encoder_hidden[0]

        encoder_hidden = encoder_hidden.view(batch_size, -1)

        z, mu, logsigma = self.vae(encoder_hidden)

        decoder_hidden = self.vae.latent_to_hidden(z)
        decoder_hidden = decoder_hidden.view(self.num_layers, batch_size, self.hidden_dim)

        # Ensure hidden state is a tuple for LSTM
        decoder_hidden = (decoder_hidden, decoder_hidden)

        embeddings = self.apply_word_dropout(input_seq)

        embeddings = self.embedding_layer(embeddings)
        logits, decoder_hidden = self.decoder(embeddings, decoder_hidden)

        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, mu, logsigma, z

    def apply_word_dropout(self, input_seq):
        if self.word_dropout_rate > 0:
            drop_mask = torch.rand(input_seq.shape).to(self.device) < self.word_dropout_rate
            drop_mask[input_seq == self.special_tokens['sos_token']] = 0
            drop_mask[input_seq == self.special_tokens['pad_token']] = 0
            input_seq = input_seq.clone()
            input_seq[drop_mask] = self.special_tokens['unk_token']
        return input_seq

    def inference(self, n=1, z=None, beam_width=5):
        if z is None:
            z = torch.randn([n, self.latent_dim]).to(self.device)
        
        decoder_hidden = self.vae.latent_to_hidden(z)
        decoder_hidden = decoder_hidden.view(self.num_layers, n, self.hidden_dim)

        # Ensure hidden state is a tuple for LSTM
        decoder_hidden = (decoder_hidden, torch.zeros_like(decoder_hidden))

        generations = []
        for i in range(n):
            generation = self.beam_search((decoder_hidden[0][:, i:i+1, :], decoder_hidden[1][:, i:i+1, :]), beam_width)
            generations.append(generation)

        return generations, z
    
    def beam_search(self, decoder_hidden, beam_width):
        sequences = [[([self.special_tokens['sos_token']], 1.0, decoder_hidden)]]

        for _ in range(self.max_gen_len):
            all_candidates = []
            for seq, score, (h, c) in sequences[-1]:
                if seq[-1] == self.special_tokens['eos_token']:
                    all_candidates.append((seq, score, (h, c)))
                    continue

                input_sequence = torch.tensor([seq[-1]]).unsqueeze(0).to(self.device)
                input_embedding = self.embedding_layer(input_sequence)
                logits, (h, c) = self.decoder(input_embedding, (h, c))
                logits = logits.squeeze(1)
                topk_probs, topk_idx = torch.topk(F.log_softmax(logits, dim=-1), beam_width)

                for k in range(beam_width):
                    candidate_seq = seq + [topk_idx[0, k].item()]
                    candidate_score = score * topk_probs[0, k].item()
                    candidate_hidden = (h.clone(), c.clone())
                    all_candidates.append((candidate_seq, candidate_score, candidate_hidden))

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            sequences.append(ordered)

        return [seq for seq, _, _ in sequences[-1]]

    def _sample(self, dist, mode='greedy'):
        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.reshape(-1)
        return sample
