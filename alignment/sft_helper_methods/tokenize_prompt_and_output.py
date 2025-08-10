from typing import List, Dict
import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(prompt_strs: List[str],
                               output_strs: List[str],
                               tokenizer: PreTrainedTokenizerBase
                               )-> Dict[str, torch.Tensor]:
    enc_prompt = tokenizer(prompt_strs, add_special_tokens=False)
    enc_output = tokenizer(output_strs, add_special_tokens=False)

    prompt_ids_batch = enc_prompt["input_ids"]
    output_ids_batch = enc_output["input_ids"]

    concat_batch = [p + o for p, o in zip(prompt_ids_batch, output_ids_batch)]

    merged_input_ids = [seq[:-1] for seq in concat_batch]
    merged_labels    = [seq[1:]  for seq in concat_batch]

    response_mask = [[0] * (len(a) - 1) + b for a, b in zip(enc_prompt["attention_mask"], enc_output["attention_mask"])]

    padded_input = tokenizer.pad(
        {
            "input_ids": merged_input_ids,
        },
        padding=True,
        return_tensors="pt"
    )

    padded_labels = tokenizer.pad(
        {
            "input_ids": merged_labels,
            "attention_mask": response_mask
        },
        padding=True,
        return_tensors="pt"
    )

    return {
        "input_ids": padded_input["input_ids"],
        "labels": padded_labels["input_ids"],
        "response_mask": padded_labels["attention_mask"]
    }