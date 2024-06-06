import torch
from torch.utils.data import Dataset, DataLoader
import json

class PTBDataset(Dataset):
    def __init__(self, data_path, vocab_path):
        try:
            with open(data_path, 'r') as file:
                self.data = json.load(file)
            with open(vocab_path, 'r') as vocab_file:
                vocab_dict = json.load(vocab_file)

            self.vocab = vocab_dict['stoi']
            self.inv_vocab = vocab_dict['itos']
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input': torch.tensor(item['input']),
            'target': torch.tensor(item['target']),
            'length': item['length']
        }
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    @property
    def pad_idx(self):
        return self.vocab['<pad>']
    
    @property
    def sos_idx(self):
        return self.vocab['<sos>']
    
    @property
    def eos_idx(self):
        return self.vocab['<eos>']
    
    @property
    def unk_idx(self):
        return self.vocab['<unk>']
    
    def get_vocab(self):
        return self.vocab
    
    def get_inv_vocab(self):
        return self.inv_vocab

def collate_fn(batch, pad_idx):
    inputs = [item['input'] for item in batch]
    targets = [item['target'] for item in batch]
    lengths = [item['length'] for item in batch]

    max_input_len = max(len(input) for input in inputs)
    max_target_len = max(len(target) for target in targets)

    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    return {
        'input': padded_inputs,
        'target': padded_targets,
        'length': torch.tensor(lengths)
    }

def create_dataloader(dataset, batch_size, pad_idx, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: collate_fn(x, pad_idx))
