import torch
from torch.utils.data import Dataset, DataLoader
import json

class PTBDataset(Dataset):
    def __init__(self, data_path, vocab_path):
        with open(data_path, 'r') as file:
            self.data = json.load(file)
        with open(vocab_path, 'r') as vocab_file:
            vocab_dict = json.load(vocab_file)
        
        self.vocab = vocab_dict['stoi']
        self.inv_vocab = vocab_dict['itos']
    
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

    # Sort by lengths (descending order)
    lengths, sorted_idx = torch.sort(torch.tensor(lengths), descending=True)
    inputs = [inputs[i] for i in sorted_idx]
    targets = [targets[i] for i in sorted_idx]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    return {
        'input': padded_inputs,
        'target': padded_targets,
        'length': lengths
    }

def create_dataloader(dataset, batch_size, pad_idx, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: collate_fn(x, pad_idx))
