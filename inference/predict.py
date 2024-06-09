import os
import argparse
import yaml
import torch
from tqdm import tqdm
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
        z = torch.randn([1, model.latent_dim]).to(device)  # Generate one latent vector at a time
        generated_tokens = model.inference(n_samples=n_samples, z=z, beam_width=beam_width, use_beam_search=use_beam_search)
        return generated_tokens

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_generated_tokens(generated_tokens, vocab, special_tokens):
    try:
        vocab_size = len(vocab['itos'])
        sentence_words = []
        for token in generated_tokens:
            if token < vocab_size:
                word = vocab['itos'][token]
                sentence_words.append(word)
            else:
                print(f"Debug: Token {token} is out of range, using <unk>")
                sentence_words.append('<unk>')

        sentence_text = ' '.join(sentence_words)
        print("Debug: Sentence text after token conversion:", sentence_text)

        return sentence_text
    except Exception as e:
        print(f"Error processing sentence {generated_tokens}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Inference with SentenceFromSpaceModel')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml', help='Path to model config file')
    parser.add_argument('--inference_config', type=str, default='configs/inference_config.yaml', help='Path to inference config file')
    parser.add_argument('--model_path', type=str, default='logs/best_model.pt', help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='raw_data', help='Directory containing the data')
    parser.add_argument('--output_file', type=str, default='generated_sentences.txt', help='File to save generated sentences')
    parser.add_argument('--use_beam_search', action='store_true', help='Use beam search for sentence generation')
    parser.add_argument('--beam_width', type=int, default=5, help='Beam width for beam search')
    args = parser.parse_args()

    model_config = load_config(args.model_config)
    inference_config = load_config(args.inference_config)

    vocab_path = os.path.join(args.data_dir, "vocab.json")
    vocab = load_vocab(vocab_path)
    special_tokens = {
        'pad_token': vocab['stoi']['<pad>'],
        'unk_token': vocab['stoi']['<unk>'],
        'sos_token': vocab['stoi']['<sos>'],
        'eos_token': vocab['stoi']['<eos>']
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, model_config, len(vocab['itos']), special_tokens, device)

    try:
        with open(args.output_file, 'w') as f:
            print(f"Opened output file: {args.output_file}")

            for _ in tqdm(range(inference_config['num_sentences']), desc="Generating Sentences"):
                sentence_tokens = generate_sentence(
                    model, 
                    device, 
                    n_samples=inference_config['n_samples'],
                    use_beam_search=args.use_beam_search,
                    beam_width=args.beam_width
                )

                sentence_text = process_generated_tokens(sentence_tokens, vocab, special_tokens)
                if sentence_text:
                    print("Generated sentence:", sentence_text)
                    f.write(sentence_text + '\n')
                else:
                    print("Generated sentence is empty or invalid, skipping writing.")

        print(f"Completed writing to {args.output_file}")

    except IOError as e:
        print(f"Failed to open or write to file {args.output_file}: {e}")

if __name__ == "__main__":
    main()
