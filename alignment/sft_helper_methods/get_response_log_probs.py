import torch
from typing import Dict
from transformers import PreTrainedModel
import torch.nn.functional as F
from alignment.sft_helper_methods.compute_entropy import compute_entropy


def get_response_log_probs(model: PreTrainedModel,
                           input_ids: torch.Tensor,
                           labels: torch.Tensor,
                           return_token_entropy: bool = False
                           ) -> Dict[str, torch.Tensor]:
    response_probs = {}
    logits = model(input_ids).logits

    log_soft = F.log_softmax(logits, dim=-1)
    response_probs["log_probs"] = log_soft.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        response_probs["token_entropy"] = compute_entropy(logits)

    return response_probs