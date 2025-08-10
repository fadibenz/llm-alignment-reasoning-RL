import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_Z = torch.logsumexp(logits, dim=-1)
    log_pi = logits - log_Z.unsqueeze(-1)
    pi = torch.exp(log_pi)

    return - torch.sum(pi * log_pi, dim=-1)