import os
import json
import pytest
from data.create_vocab import create_vocab, load_vocab

def test_create_vocab():
    data_path = "test_data/ptb.train.txt"
    vocab_path = "test_data/ptb.vocab.json"

    # Create test data
    os.makedirs("test_data", exist_ok=True)
    with open(data_path, 'w') as file:
        file.write("This is a test sentence.\n")
        file.write("This is a test sentence.\n")
        file.write("This is a sentence.\n") 

    create_vocab(data_path, vocab_path, min_freq=2)
    assert os.path.exists(vocab_path)

    vocab = load_vocab(vocab_path)
    print(vocab.get_itos())
    assert len(vocab.get_itos()) == 10
    assert vocab['<pad>'] == 0
    assert vocab['<unk>'] == 1
    assert vocab['<sos>'] == 2
    assert vocab['<eos>'] == 3

    os.remove(data_path)
    os.remove(vocab_path)
    os.rmdir("test_data")

if __name__ == "__main__":
    pytest.main([__file__])
