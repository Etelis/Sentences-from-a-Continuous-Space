import os
import argparse
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.model import SentenceFromSpaceModel
from data.dataset import PTBDataset, create_dataloader
from data.create_vocab import create_vocab, load_vocab
from data.download_data import download_and_extract_data
from data.preprocess_data import preprocess_data, save_preprocessed_data
from training.loss_functions import loss_fn
from training.metrics import calculate_accuracy
from training.evaluate import evaluate

torch.autograd.set_detect_anomaly(True)

def train(model, train_dataloader, val_dataloader, optimizer, device, epochs, log_interval, log_dir, anneal_function, k, annealing_till):
    writer = SummaryWriter(log_dir)
    
    model.to(device)
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_kl_div = 0
        total_recon_loss = 0
        total_accuracy = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            inputs, targets, lengths = batch['input'], batch['target'], batch['length']
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Ensure lengths are sorted in descending order
            lengths, sorted_idx = torch.sort(lengths, descending=True)
            inputs, targets = inputs[sorted_idx], targets[sorted_idx]

            optimizer.zero_grad()
            outputs, mu, logsigma, _ = model(inputs, lengths)
            elbo, nll_loss, kl_loss, kl_weight = loss_fn(outputs, targets, lengths, mu, logsigma, anneal_function, global_step, k, annealing_till)
            elbo.backward()
            optimizer.step()
            
            total_loss += elbo.item()
            total_kl_div += kl_loss.item()
            total_recon_loss += nll_loss.item()
            accuracy = calculate_accuracy(outputs, targets, model.special_tokens['pad_token'])
            total_accuracy += accuracy

            if batch_idx % log_interval == 0:
                writer.add_scalar('Train/ELBO', elbo.item(), global_step)
                writer.add_scalar('Train/KL Divergence', kl_loss.item(), global_step)
                writer.add_scalar('Train/Negative Log-likelihood', nll_loss.item(), global_step)
                writer.add_scalar('Train/KL Weight', kl_weight, global_step)
                writer.add_scalar('Train/Accuracy', accuracy, global_step)

            global_step += 1

        avg_train_loss = total_loss / len(train_dataloader)
        avg_kl_div = total_kl_div / len(train_dataloader)
        avg_recon_loss = total_recon_loss / len(train_dataloader)
        avg_train_accuracy = total_accuracy / len(train_dataloader)
        val_loss = evaluate(model, val_dataloader, loss_fn, device)

        writer.add_scalar('Epoch/ELBO', avg_train_loss, epoch)
        writer.add_scalar('Epoch/KL Divergence', avg_kl_div, epoch)
        writer.add_scalar('Epoch/Negative Log-likelihood', avg_recon_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', avg_train_accuracy, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)

        print(f'Epoch: {epoch+1}, Train ELBO: {avg_train_loss:.4f}, KL Divergence: {avg_kl_div:.4f}, Recon Loss: {avg_recon_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Validation Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))
    
    writer.close()

def load_config(config_file):
    with open(config_file, 'r') as file, open(file.name.replace(".yaml", "_copy.yaml"), 'w') as out_file:
        config = yaml.safe_load(file)
        yaml.safe_dump(config, out_file)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SentenceFromSpaceModel')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml', help='Path to model config file')
    parser.add_argument('--training_config', type=str, default='configs/training_config.yaml', help='Path to training config file')
    parser.add_argument('--data_url', type=str, default="http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz", help='URL for dataset download')
    parser.add_argument('--data_dir', type=str, default='raw_data', help='Directory for data storage')
    args = parser.parse_args()

    model_config = load_config(args.model_config)
    training_config = load_config(args.training_config)
    
    data_url = args.data_url
    data_dir = args.data_dir
    vocab_path = os.path.join(data_dir, "vocab.json")
    train_data_path = os.path.join(data_dir, "ptb.train.json")
    val_data_path = os.path.join(data_dir, "ptb.valid.json")
    
    # Download and preprocess data
    if not os.path.exists(data_dir):
        download_and_extract_data(data_url, data_dir)
    
    if not os.path.exists(vocab_path):
        vocab = create_vocab(os.path.join(data_dir, "ptb.train.txt"), vocab_path)
        vocab = {'stoi': vocab.get_stoi(), 'itos': vocab.get_itos()}
    else:
        vocab = load_vocab(vocab_path)

    if not os.path.exists(train_data_path):
        train_data = preprocess_data(os.path.join(data_dir, "ptb.train.txt"), vocab['stoi'])
        save_preprocessed_data(train_data, train_data_path)

    if not os.path.exists(val_data_path):
        val_data = preprocess_data(os.path.join(data_dir, "ptb.valid.txt"), vocab['stoi'])
        save_preprocessed_data(val_data, val_data_path)
    
    # Create datasets and dataloaders
    train_dataset = PTBDataset(train_data_path, vocab_path)
    val_dataset = PTBDataset(val_data_path, vocab_path)
    train_dataloader = create_dataloader(train_dataset, batch_size=training_config['batch_size'], pad_idx=vocab['stoi']['<pad>'], shuffle=True)
    val_dataloader = create_dataloader(val_dataset, batch_size=training_config['batch_size'], pad_idx=vocab['stoi']['<pad>'], shuffle=False)
    
    # Define model, criterion, optimizer
    special_tokens = {'pad_token': vocab['stoi']['<pad>'], 'unk_token': vocab['stoi']['<unk>'], 'sos_token': vocab['stoi']['<sos>'], 'eos_token': vocab['stoi']['<eos>']}
    model = SentenceFromSpaceModel(vocab_size=len(vocab['stoi']), 
                                   embedding_dim=model_config['embedding_dim'], 
                                   hidden_dim=model_config['hidden_dim'], 
                                   word_dropout_rate=model_config['word_dropout_rate'],
                                   embedding_dropout_rate=model_config['embedding_dropout_rate'], 
                                   latent_dim=model_config['latent_dim'], 
                                   special_tokens=special_tokens,
                                   max_gen_len=model_config['max_gen_len'], 
                                   bidirectional=model_config['bidirectional'], 
                                   num_layers=model_config['num_layers'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    train(model, train_dataloader, val_dataloader, optimizer, device, 
          epochs=training_config['epochs'], 
          log_interval=training_config['log_interval'], 
          log_dir=training_config['log_dir'], 
          anneal_function=training_config['anneal_function'], 
          k=training_config['k'], 
          annealing_till=training_config['annealing_till'])