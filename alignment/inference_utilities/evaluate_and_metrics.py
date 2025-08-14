import math
import statistics
import random
from typing import Callable, Dict, List
from vllm import SamplingParams
import torch

def evaluate_and_metrics(
        vllm_model,
        reward_fn: Callable[[str, str], Dict[str, float]],
        prompts: List[str],
        ground_truths: List[str],
        vllm_device: str,
        sampling_params: SamplingParams,
        sample_size: int = 5
) -> Dict[str, any]:

    with torch.cuda.device(vllm_device):
        outputs = vllm_model.generate(prompts, sampling_params=sampling_params)

    answer_scores, format_scores, lengths, entropies = [], [], [], []
    lengths_correct, lengths_incorrect = [], []
    entropies_correct, entropies_incorrect = [], []
    samples = []

    for output, prompt, ground_truth in zip(outputs, prompts, ground_truths):
        text = output.outputs[0].text if output.outputs else ""
        scores = reward_fn(text, ground_truth)

        answer_reward = scores.get("answer_reward", 0)
        format_reward = scores.get("format_reward", 0)
        length = len(text.split())
        entropy = calculate_entropy(output.outputs[0])

        answer_scores.append(answer_reward)
        format_scores.append(format_reward)
        lengths.append(length)
        entropies.append(entropy)

        if answer_reward == 1:
            lengths_correct.append(length)
            entropies_correct.append(entropy)
        else:
            lengths_incorrect.append(length)
            entropies_incorrect.append(entropy)

        samples.append((prompt, text, ground_truth, answer_reward, format_reward, length, entropy))

    return {
        "answer_reward_mean": statistics.mean(answer_scores),
        "format_reward_mean": statistics.mean(format_scores),

        "avg_response_length": statistics.mean(lengths),
        "avg_length_correct": statistics.mean(lengths_correct) if lengths_correct else 0,
        "avg_length_incorrect": statistics.mean(lengths_incorrect) if lengths_incorrect else 0,

        "avg_token_entropy": statistics.mean(entropies),
        "avg_entropy_correct": statistics.mean(entropies_correct) if entropies_correct else 0,
        "avg_entropy_incorrect": statistics.mean(entropies_incorrect) if entropies_incorrect else 0,

        "random_samples": random.sample(samples, min(sample_size, len(samples)))
    }


def calculate_entropy(output) -> float:
    if not hasattr(output, 'logprobs') or not output.logprobs:
        return 0.0

    entropies = []
    for token_logprobs in output.logprobs:
        if not token_logprobs:
            continue
        logprobs = list(token_logprobs.values())
        probs = [math.exp(lp) for lp in logprobs]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        entropies.append(entropy)

    return statistics.mean(entropies) if entropies else 0.0