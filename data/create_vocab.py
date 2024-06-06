import spacy
import torchtext; torchtext.disable_torchtext_deprecation_warning()
from collections import Counter, OrderedDict
import json
from torchtext.vocab import vocab as Vocab
from torchtext.data.utils import get_tokenizer
import os

def create_vocab(data_path, vocab_path, min_freq=3):
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    counter = Counter()
    with open(data_path, 'r') as file:
        for line in file:
            tokens = tokenizer(line.strip())
            counter.update(tokens)

    filtered_counter = {token: count for token, count in counter.items() if count >= min_freq}
    ordered_dict = OrderedDict(sorted(filtered_counter.items(), key=lambda x: x[1], reverse=True))

    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    v = Vocab(OrderedDict([(token, 1) for token in special_tokens]), specials=special_tokens)
    for token in ordered_dict:
        if token not in v:
            v.append_token(token)
    v.set_default_index(v['<unk>'])

    vocab_dict = {
        'stoi': v.get_stoi(),
        'itos': v.get_itos()
    }
    
    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    
    return v

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as vocab_file:
        vocab_dict = json.load(vocab_file)
    
    ordered_dict = OrderedDict([(token, 1) for token in vocab_dict['itos']])
    v = Vocab(ordered_dict, specials=['<pad>', '<unk>', '<sos>', '<eos>'])
    v.set_default_index(v['<unk>'])

    return vocab_dict
