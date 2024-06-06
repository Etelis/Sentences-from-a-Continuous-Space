import spacy
import json
from torchtext.data.utils import get_tokenizer
from multiprocessing import Pool

def preprocess_line(line, tokenizer, stoi, max_len):
    try:
        tokens = tokenizer(line.strip())
        input_seq = ['<sos>'] + tokens[:max_len]
        target_seq = tokens[:max_len] + ['<eos>']

        input_seq = [stoi.get(token, stoi['<unk>']) for token in input_seq]
        target_seq = [stoi.get(token, stoi['<unk>']) for token in target_seq]

        return {'input': input_seq, 'target': target_seq, 'length': len(tokens) + 1}
    except Exception as e:
        print(f"Error processing line: {line}")
        print(f"Exception: {e}")
        raise

def preprocess_data(data_path, stoi, max_len=50):
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    with open(data_path, 'r') as file:
        lines = file.readlines()

    with Pool() as pool:
        data = pool.starmap(preprocess_line, [(line, tokenizer, stoi, max_len) for line in lines])

    return data

def save_preprocessed_data(data, save_path):
    try:
        with open(save_path, 'w') as save_file:
            json.dump(data, save_file)
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")
        raise
