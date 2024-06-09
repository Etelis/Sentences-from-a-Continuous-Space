import os
import yaml
import torch
from models.model import SentenceFromSpaceModel
from data.create_vocab import load_vocab

def load_model(model_path, model_config, vocab_size, special_tokens, device):
    model = SentenceFromSpaceModel(
        vocab_size=vocab_size,
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        word_dropout_rate=model_config['word_dropout_rate'],
        embedding_dropout_rate=model_config['embedding_dropout_rate'],
        latent_dim=model_config['latent_dim'],
        special_tokens=special_tokens,
        max_gen_len=model_config['max_gen_len'],
        bidirectional=model_config['bidirectional'],
        num_layers=model_config['num_layers']
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def generate_sentence(model, device, n_samples, use_beam_search, beam_width):
    with torch.no_grad():
        z = torch.randn([n_samples, model.latent_dim]).to(device)
        generated_tokens_list = model.inference(n_samples=n_samples, z=z, beam_width=beam_width, use_beam_search=use_beam_search)
        if use_beam_search:
            generated_tokens_list = [tokens[0] for tokens in generated_tokens_list]

        return generated_tokens_list

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_generated_tokens(generated_tokens, vocab):
    try:
        vocab_size = len(vocab['itos'])
        sentence_words = []
        for token in generated_tokens:
            if token < vocab_size:
                word = vocab['itos'][token]
                sentence_words.append(word)
            else:
                sentence_words.append('<unk>')

        sentence_text = ' '.join(sentence_words)
        return sentence_text
    except Exception as e:
        print(f"Error processing sentence {generated_tokens}: {e}")
        return None
