from typing import Callable, Dict, List
from vllm import LLM, SamplingParams
import statistics
import random


def evaluate_and_metrics(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    sampling_params: SamplingParams,
    sample_size: int = 5
) -> Dict[str, any]:

    answer_scores = []
    format_scores = []
    lengths = []
    lengths_correct = []
    lengths_incorrect = []
    samples = []

    outputs = vllm_model.generate(prompts, sampling_params)

    for output, ground_truth in zip(outputs, ground_truths):
        generated_text = output.outputs[0].text if output.outputs else ""
        evaluation_scores = reward_fn(generated_text, ground_truth)

        answer_reward = evaluation_scores.get("answer_reward", 0)
        format_reward = evaluation_scores.get("format_reward", 0)

        answer_scores.append(answer_reward)
        format_scores.append(format_reward)

        length = len(generated_text.split())
        lengths.append(length)

        if answer_reward == 1:
            lengths_correct.append(length)
        else:
            lengths_incorrect.append(length)

        samples.append((generated_text, ground_truth))

    metrics = {
        "answer_reward_mean": statistics.mean(answer_scores) if answer_scores else 0.0,
        "format_reward_mean": statistics.mean(format_scores) if format_scores else 0.0,
        "avg_response_length": statistics.mean(lengths) if lengths else 0.0,
        "avg_length_correct": statistics.mean(lengths_correct) if lengths_correct else 0.0,
        "avg_length_incorrect": statistics.mean(lengths_incorrect) if lengths_incorrect else 0.0,
        "random_samples": random.sample(samples, k=min(sample_size, len(samples)))
    }

    return metrics