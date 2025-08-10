import torch

def masked_normalize( tensor: torch.Tensor,
                      mask: torch.Tensor,
                      normalize_constant: float,
                      dim: int | None = None,
                      ) -> torch.Tensor:

    return torch.sum(tensor * mask, dim=dim) / normalize_constant