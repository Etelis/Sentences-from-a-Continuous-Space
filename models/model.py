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


    def inference(self, n_samples, z=None, beam_width=5, use_beam_search=False):

        if z is None:
            # Generate a random latent vector if not provided
            z = torch.randn([1, self.latent_dim]).to(self.device)
    
        if use_beam_search:
            return self.beam_search(z, beam_width=beam_width, max_seq_len=n_samples)
        else:
            # Greedy decoding
            sos_token = self.special_tokens['sos_token']
            eos_token = self.special_tokens['eos_token']
            input_token = torch.tensor([[sos_token]]).to(self.device)
    
            decoder_hidden = self.vae.latent_to_hidden(z)
            decoder_hidden = decoder_hidden.view(self.num_layers, 1, self.hidden_dim)
            decoder_hidden = (decoder_hidden, torch.zeros_like(decoder_hidden))
    
            generated_tokens = []
    
            for _ in range(n_samples):
                input_embedding = self.embedding_layer(input_token)
                logits, decoder_hidden = self.decoder(input_embedding, decoder_hidden)
                log_probs = F.log_softmax(logits, dim=-1)
                top_token = torch.argmax(log_probs, dim=-1)
                generated_tokens.append(top_token.item())
    
                if top_token.item() == eos_token:
                    break
    
                input_token = top_token
    
            return generated_tokens

    
    def beam_search(self, z, beam_width=5, max_seq_len=20):
        
        sos_token = self.special_tokens['sos_token']
        eos_token = self.special_tokens['eos_token']
    
        decoder_hidden = self.vae.latent_to_hidden(z)
        decoder_hidden = decoder_hidden.view(self.num_layers, 1, self.hidden_dim)
        decoder_hidden = (decoder_hidden, torch.zeros_like(decoder_hidden))

        input_token = torch.tensor([[sos_token]]).to(self.device)
    
        sequences = [[list(), 0.0, decoder_hidden, input_token]]
        
        for _ in range(max_seq_len):
            all_candidates = list()
            for seq, score, hidden, input_token in sequences:
                if len(seq) > 0 and seq[-1] == eos_token:
                    all_candidates.append((seq, score, hidden, input_token))
                    continue
    
                input_embedding = self.embedding_layer(input_token)
                logits, hidden = self.decoder(input_embedding, hidden)
                log_probs = F.log_softmax(logits, dim=-1)
    
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
    
                for i in range(beam_width):
                    candidate = (seq + [topk_indices[0, 0, i].item()],
                                 score + topk_log_probs[0, 0, i].item(),
                                 hidden,
                                 topk_indices[0, 0, i].view(1, 1))
                    all_candidates.append(candidate)
    
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
    
            if all(seq[-1] == eos_token for seq, _, _, _ in sequences):
                break

        return sequences[0][0]