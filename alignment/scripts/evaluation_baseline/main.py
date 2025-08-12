import json
import logging
from pathlib import Path

from vllm import LLM, SamplingParams
from alignment.utils.utils import setup_logging, set_seed_everything
from alignment.scripts.evaluation_baseline.config import parse_args
from alignment.inference_utilities.evaluate_vllm import evaluate_vllm
from alignment.inference_utilities.reward_function import make_reward_fn
from alignment.utils.data import load_validation_data


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    set_seed_everything(args.seed)

    logging.info("Starting evaluation...")
    logging.info(f"Arguments: {vars(args)}")

    input_path = Path(args.dataset)
    prompt_path = Path(args.prompt_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = prompt_path.read_text(encoding="utf-8")

    prompt_list, ground_truth_list = load_validation_data(input_path, prompt_template)

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
        include_stop_str_in_output=args.include_stop_str_in_output,
        seed=args.seed
    )


    reward_fn = make_reward_fn(ground_truth_list)

    results_path = output_dir / f"results.jsonl"
    summary_path = output_dir / "summary.json"

    evaluate_vllm(vllm_model, reward_fn, prompt_list, eval_sampling_params, results_path, summary_path)