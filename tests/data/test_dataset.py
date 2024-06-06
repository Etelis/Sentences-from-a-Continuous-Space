import os
import json
import pytest
import torch
from torch.utils.data import DataLoader
from data.dataset import PTBDataset, collate_fn, create_dataloader
from data.create_vocab import create_vocab, load_vocab
from data.preprocess_data import preprocess_data, save_preprocessed_data

def test_ptb_dataset_and_dataloader():
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
    stoi = vocab['stoi']

    # Preprocess data
    preprocessed_data = preprocess_data(data_path, stoi)
    save_preprocessed_data(preprocessed_data, preprocessed_data_path)
    
    # Create dataset and dataloader
    dataset = PTBDataset(preprocessed_data_path, vocab_path)
    dataloader = create_dataloader(dataset, batch_size=2, pad_idx=dataset.pad_idx)

    # Check dataset length
    assert len(dataset) == 2

    # Check dataloader output
    for batch in dataloader:
        inputs = [item['input'] for item in dataset]
        targets = [item['target'] for item in dataset]

        max_input_len = max(len(input) for input in inputs)
        max_target_len = max(len(target) for target in targets)

        assert batch['input'].shape[0] == 2
        assert batch['input'].shape[1] == max_input_len
        assert batch['target'].shape[0] == 2
        assert batch['target'].shape[1] == max_target_len
        assert batch['length'].shape[0] == 2

    # Clean up
    os.remove(data_path)
    os.remove(vocab_path)
    os.remove(preprocessed_data_path)
    os.rmdir("test_data")

if __name__ == "__main__":
    pytest.main([__file__])
