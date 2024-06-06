import torch

def calculate_accuracy(log_probs, targets, pad_idx):
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    targets_flat = targets.view(-1)

    _, predicted = torch.max(log_probs_flat, dim=1)
    correct = (predicted == targets_flat).float()

    ignore_mask = (targets_flat != pad_idx).float()
    correct *= ignore_mask

    accuracy = correct.sum() / ignore_mask.sum()
    return accuracy.item()
