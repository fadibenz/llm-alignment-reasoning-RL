from vllm import LLM, SamplingParams
from typing import Callable, Dict, List
import json
from pathlib import Path

def evaluate_vllm(
        vllm_model: LLM,
        reward_fn:  Callable[[str, int], Dict[str, float]],
        prompts: List[str],
        eval_sampling_params: SamplingParams,
        file_path: str | Path
    ) -> None:

    outputs = vllm_model.generate(prompts, eval_sampling_params)
    to_serialize_output = []
    number_correct_answers = 0
    number_correctly_formatted = 0

    for idx, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        evaluation_scores = reward_fn(generated_text, idx)
        number_correct_answers += evaluation_scores["answer_reward"]
        number_correctly_formatted += evaluation_scores["format_reward"]
        to_serialize_output.append({
            "prompt": prompt,
            "output": generated_text,
            "evaluation_scores": evaluation_scores
        })

    with open(file_path, "w", encoding="utf-8") as f:

        for record in to_serialize_output:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        f.write(json.dumps({
            "number_correct_answers": number_correct_answers,
            "number_correctly_formatted": number_correctly_formatted
        }))