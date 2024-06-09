import os
import argparse
import torch
from tqdm import tqdm
from sentence_generator import load_model, generate_sentence, load_config, process_generated_tokens
from data.create_vocab import load_vocab

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
                generated_tokens_list = generate_sentence(
                    model, 
                    device, 
                    n_samples=1,  # Generate one sample at a time for now
                    use_beam_search=args.use_beam_search,
                    beam_width=args.beam_width
                )

                for generated_tokens in generated_tokens_list:
                    sentence_text = process_generated_tokens(generated_tokens, vocab)
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
