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

        _, encoder_hidden = self.encoder(embeddings, seq_lengths)
        if isinstance(encoder_hidden, tuple):  # LSTM case
            encoder_hidden = encoder_hidden[0]

        encoder_hidden = encoder_hidden.view(batch_size, -1)

        z, mu, logsigma = self.vae(encoder_hidden)

        decoder_hidden = self.vae.latent_to_hidden(z)
        decoder_hidden = decoder_hidden.view(self.num_layers, batch_size, self.hidden_dim)

        # Ensure hidden state is a tuple for LSTM
        decoder_hidden = (decoder_hidden, decoder_hidden)

        embeddings = self.apply_word_dropout(input_seq)

        logits, decoder_hidden = self.decoder(embeddings, decoder_hidden)

        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, mu, logsigma, z

    def apply_word_dropout(self, input_seq):
        if self.word_dropout_rate > 0:
            prob = torch.rand(input_seq.shape).to(self.device)
            input_seq_with_dropout = input_seq.clone()
            prob[input_seq_with_dropout == self.special_tokens['sos_token']] = 1
            prob[input_seq_with_dropout == self.special_tokens['pad_token']] = 1
            input_seq_with_dropout[prob < self.word_dropout_rate] = self.special_tokens['unk_token']
            embeddings = self.embedding_layer(input_seq_with_dropout)
            return embeddings
        return self.embedding_layer(input_seq)


    def inference(self, n_samples=1, z=None, use_beam_search=True, beam_width=5):
        if z is None:
            z = torch.randn([n_samples, self.latent_dim]).to(self.device)
    
        decoder_hidden = self.vae.latent_to_hidden(z)
        decoder_hidden = decoder_hidden.view(self.num_layers, n_samples, self.hidden_dim)
        decoder_hidden = (decoder_hidden, torch.zeros_like(decoder_hidden))
    
        if use_beam_search:
            generations = [self.beam_search(decoder_hidden, beam_width) for _ in range(n_samples)]
        else:
            generations = [self.greedy_search(decoder_hidden) for _ in range(n_samples)]
    
        return generations
    
    def beam_search(self, decoder_hidden, beam_width):
        sequences = [[([], 0.0, decoder_hidden)]]
        for _ in range(self.max_gen_len):
            all_candidates = []
            for seq, score, (h, c) in sequences[-1]:
                if len(seq) > 0 and seq[-1] == self.special_tokens['eos_token']:
                    all_candidates.append((seq, score, (h, c)))
                    continue
    
                input_token = torch.tensor([[seq[-1]]] if seq else [[self.special_tokens['sos_token']]]).to(self.device)
                input_embedding = self.embedding_layer(input_token)
                logits, (h, c) = self.decoder(input_embedding, (h, c))
                log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
    
                topk_probs, topk_indices = torch.topk(log_probs, beam_width)
                for k in range(beam_width):
                    candidate_seq = seq + [topk_indices[0, k].item()]
                    candidate_score = score + topk_probs[0, k].item() / (len(candidate_seq) ** 0.7)  # Length normalization
                    all_candidates.append((candidate_seq, candidate_score, (h.clone(), c.clone())))
    
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            sequences.append(ordered)
    
        final_sequences = sorted(sequences[-1], key=lambda x: x[1], reverse=True)
        return final_sequences[0][0]


    def greedy_search(self, decoder_hidden):
        generated_sequence = []
        input_token = torch.tensor([[self.special_tokens['sos_token']]]).to(self.device)
        for _ in range(self.max_gen_len):
            input_embedding = self.embedding_layer(input_token)
            logits, decoder_hidden = self.decoder(input_embedding, decoder_hidden)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
            top_token = log_probs.argmax(dim=-1).item()
            generated_sequence.append(top_token)
            if top_token == self.special_tokens['eos_token']:
                break
            input_token = torch.tensor([[top_token]]).to(self.device)

        return generated_sequence