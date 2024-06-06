import os
import json
import pytest
from data.preprocess_data import preprocess_data, save_preprocessed_data
from data.create_vocab import create_vocab, load_vocab

def test_preprocess_data():
    data_path = "test_data/ptb.train.txt"
    vocab_path = "test_data/ptb.vocab.json"
    preprocessed_data_path = "test_data/ptb.train.preprocessed.json"

    # Create test data
    os.makedirs("test_data", exist_ok=True)
    with open(data_path, 'w') as file:
        file.write("This is a test sentence.\n")
        file.write("This is another test sentence.\n")

    # Create vocabulary
    create_vocab(data_path, vocab_path, min_freq=1)
    vocab = load_vocab(vocab_path)

    if 'stoi' in vocab and isinstance(vocab['stoi'], dict):
        stoi = vocab['stoi']
    else:
        print("Error: `stoi` not found or is not a dictionary.")
        return

    preprocessed_data = preprocess_data(data_path, stoi)
    save_preprocessed_data(preprocessed_data, preprocessed_data_path)

    assert os.path.exists(preprocessed_data_path)

    with open(preprocessed_data_path, 'r') as file:
        data = json.load(file)
    assert len(data) == 2
    assert data[0]['length'] == 7
    assert data[0]['input'][0] == stoi['<sos>']
    assert data[0]['target'][-1] == stoi['<eos>']

    # Clean up
    os.remove(data_path)
    os.remove(vocab_path)
    os.remove(preprocessed_data_path)
    os.rmdir("test_data")

if __name__ == "__main__":
    pytest.main([__file__])
