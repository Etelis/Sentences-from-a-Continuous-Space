import os
import argparse
import yaml
import torch
from tqdm import tqdm
from models.model import SentenceFromSpaceModel
from data.dataset import PTBDataset, create_dataloader
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

def generate_sentences(model, dataloader, device, beam_width=5):
    all_generations = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Sentences"):
            lengths = batch['length']
            z = torch.randn([len(lengths), model.latent_dim]).to(device)
            generations, _ = model.inference(n=len(lengths), z=z, beam_width=beam_width)
            all_generations.extend(generations)
    return all_generations

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description='Inference with SentenceFromSpaceModel')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml', help='Path to model config file')
    parser.add_argument('--inference_config', type=str, default='configs/inference_config.yaml', help='Path to inference config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the data')
    parser.add_argument('--output_file', type=str, default='generated_sentences.txt', help='File to save generated sentences')
    args = parser.parse_args()

    model_config = load_config(args.model_config)
    inference_config = load_config(args.inference_config)

    data_dir = args.data_dir
    vocab_path = os.path.join(data_dir, "vocab.json")
    test_data_path = os.path.join(data_dir, "ptb.test.json")

    vocab = load_vocab(vocab_path)
    special_tokens = {
        'pad_token': vocab['stoi']['<pad>'],
        'unk_token': vocab['stoi']['<unk>'],
        'sos_token': vocab['stoi']['<sos>'],
        'eos_token': vocab['stoi']['<eos>']
    }

    test_dataset = PTBDataset(test_data_path, vocab_path)
    test_dataloader = create_dataloader(test_dataset, batch_size=inference_config['batch_size'], pad_idx=vocab['stoi']['<pad>'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, model_config, len(vocab['stoi']), special_tokens, device)

    generations = generate_sentences(model, test_dataloader, device, beam_width=inference_config['beam_width'])

    with open(args.output_file, 'w') as f:
        for sentence in generations:
            sentence_text = ' '.join([vocab['itos'][token] for token in sentence])
            f.write(sentence_text + '\n')

if __name__ == "__main__":
    main()
