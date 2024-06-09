import torch

def evaluate(model, dataloader, criterion, device, anneal_function, step, k, annealing_till):
    model.eval()
    total_loss = 0
    total_nll_loss = 0
    total_kl_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, lengths = batch['input'], batch['target'], batch['length']
            inputs, targets = inputs.to(device), targets.to(device)

            outputs, mu, logsigma, _ = model(inputs, lengths)
            loss, nll_loss, kl_loss, kl_weight = criterion(outputs, targets, lengths, mu, logsigma, anneal_function, step, k, annealing_till)
            total_loss += loss.item()
            total_nll_loss += nll_loss.item()
            total_kl_loss += kl_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_nll_loss = total_nll_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)
    
    return avg_loss, avg_nll_loss, avg_kl_loss