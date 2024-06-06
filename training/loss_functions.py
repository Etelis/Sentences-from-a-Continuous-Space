import torch
import torch.nn.functional as F

def loss_nll(logp, target):
    return F.nll_loss(logp, target, reduction='sum')

def kl_anneal_function(anneal_function, step, k, annealing_till):
    if anneal_function == 'logistic':
        return float(1 / (1 + torch.exp(torch.tensor(-k * (step - annealing_till)))))
    elif anneal_function == 'linear':
        return min(1, step / annealing_till)
    else:
        raise ValueError("Invalid anneal function")

def loss_fn(logp, target, length, mean, log_sigmasquared, anneal_function, step, k, annealing_till):
    target = target[:, :torch.max(length).item()].contiguous().view(-1)
    logp = logp.view(-1, logp.size(2))

    # Negative Log Likelihood
    nll_loss = loss_nll(logp, target)

    # KL Divergence
    kl_loss = -0.5 * torch.sum(1 + log_sigmasquared - mean**2 - torch.exp(log_sigmasquared))
    kl_weight = kl_anneal_function(anneal_function, step, k, annealing_till)

    elbo = (nll_loss + kl_weight * kl_loss) / logp.size(0)  # Normalizing by the number of elements
    return elbo, nll_loss, kl_loss, kl_weight
