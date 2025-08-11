import json
import logging
from pathlib import Path

from vllm import LLM, SamplingParams
from alignment.utils.utils import setup_logging
from alignment.scripts.evaluation_baseline.config import parse_args
from alignment.utils.drgrpo_grader import r1_zero_reward_fn
from alignment.inference_utilities.evaluate_vllm import evaluate_vllm


def load_prompts_and_answers(input_path: Path, prompt_template: str):
    prompts, answers = [], []

    if input_path.suffix != ".jsonl":
        raise ValueError("Input file must be in .jsonl format.")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            json_obj = json.loads(line)
            if "{question}" not in prompt_template:
                raise ValueError("Prompt template must contain '{question}'.")
            prompt = prompt_template.replace("{question}", json_obj["problem"])
            prompts.append(prompt)
            answers.append(json_obj["answer"])

    logging.info(f"Loaded {len(prompts)} prompts.")
    return prompts, answers


def make_reward_fn(ground_truths):
    def reward_fn(output_text, idx):
        ref = ground_truths[idx]
        return r1_zero_reward_fn(output_text, ref)
    return reward_fn


if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    logging.info("Starting evaluation...")
    logging.info(f"Arguments: {vars(args)}")

    input_path = Path(args.dataset)
    prompt_path = Path(args.prompt_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = prompt_path.read_text(encoding="utf-8")

    prompt_list, ground_truth_list = load_prompts_and_answers(input_path, prompt_template)

    logging.info(f"Loading model: {args.model_name}")

    vllm_model = LLM(
        model=args.model_name,
        max_num_batched_tokens=args.max_batched_tokens
    )

    eval_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop_sequences,
        include_stop_str_in_output=getattr(args, "include_stop_str_in_output", True)
    )


    reward_fn = make_reward_fn(ground_truth_list)

    results_path = output_dir / f"results.jsonl"
    summary_path = output_dir / "summary.json"

    evaluate_vllm(vllm_model, reward_fn, prompt_list, eval_sampling_params, results_path, summary_path)