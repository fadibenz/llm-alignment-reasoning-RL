import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Union
from vllm import LLM, SamplingParams


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, int], Dict[str, float]],
    prompts: List[str],
    sampling_params: SamplingParams,
    results_path: Union[str, Path],
    summary_path: Union[str, Path]
) -> None:

    total_correct_answers = 0
    total_correct_formatting = 0

    with open(results_path, "w", encoding="utf-8") as results_file:
        outputs = vllm_model.generate(prompts, sampling_params=sampling_params)

        for idx, output in enumerate(outputs):
            generated_text = output.outputs[0].text if output.outputs else ""
            evaluation_scores = reward_fn(generated_text, idx)

            total_correct_answers += evaluation_scores.get("answer_reward", 0)
            total_correct_formatting += evaluation_scores.get("format_reward", 0)

            json.dump({
                "prompt": output.prompt,
                "output": generated_text,
                "evaluation_scores": evaluation_scores
            }, results_file, ensure_ascii=False)
            results_file.write("\n")

    summary = {
        "total_correct_answers": total_correct_answers,
        "total_correct_formatting": total_correct_formatting,
        "total_number": len(prompts)
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info(f"Evaluation complete. Results in {results_path}, summary in {summary_path}")
