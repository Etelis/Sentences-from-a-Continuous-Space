import pytest
import torch
from models.model import SentenceFromSpaceModel

@pytest.fixture
def sentence_vae():
    special_tokens = {'pad_token': 0, 'unk_token': 1, 'sos_token': 2, 'eos_token': 3}
    return SentenceFromSpaceModel(vocab_size=100, embedding_dim=10, hidden_dim=20, word_dropout_rate=0.5,
                            embedding_dropout_rate=0.1, latent_dim=15, special_tokens=special_tokens,
                            max_gen_len=50, bidirectional=False, num_layers=2)

def test_sentence_vae_forward(sentence_vae):
    input_seq = torch.randint(0, 100, (5, 7))  # batch_size=5, seq_len=7
    seq_lengths = torch.tensor([7, 6, 5, 4, 3])
    
    log_probs, mu, logsigma, z = sentence_vae(input_seq, seq_lengths)
    
    assert log_probs.shape == (5, 7, 100)  # (batch_size, seq_len, vocab_size)
    assert mu.shape == (5, 15)  # (batch_size, latent_dim)
    assert logsigma.shape == (5, 15)  # (batch_size, latent_dim)
    assert z.shape == (5, 15)  # (batch_size, latent_dim)

def test_sentence_vae_inference(sentence_vae):
    generations, z = sentence_vae.inference(n=5, beam_width=3)
    
    assert len(generations) == 5  # Ensure 5 sequences are generated
    assert all(len(seq) <= sentence_vae.max_gen_len for seq in generations)  # Ensure sequences do not exceed max_gen_len
    assert z.shape == (5, 15)  # Ensure latent vector shape is correct

    for i, gen in enumerate(generations):
        print(f"Generation {i+1}: {gen}")
