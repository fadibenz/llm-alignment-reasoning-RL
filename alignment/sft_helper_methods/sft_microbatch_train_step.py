import torch

from alignment.sft_helper_methods.masked_normalize import masked_normalize


def sft_microbatch_train_step( policy_log_probs: torch.Tensor,
                               response_mask: torch.Tensor,
                               gradient_accumulation_steps: int,
                               normalize_constant: float = 1.0,
                               ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    denominator = gradient_accumulation_steps * policy_log_probs.size(0)

    loss = - masked_normalize(policy_log_probs,
                              response_mask,
                              normalize_constant) / denominator

    metadata = {}

    return loss, metadata