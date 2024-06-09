import torch

def evaluate(model, dataloader, criterion, device, anneal_function, step, k, annealing_till):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, lengths = batch['input'], batch['target'], batch['length']
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, mu, logsigma, _ = model(inputs, lengths)
            loss, nll_loss, kl_loss, kl_weight = criterion(outputs, targets, lengths, mu, logsigma, anneal_function, step, k, annealing_till)
            total_loss += loss.item()

    return total_loss / len(dataloader)
