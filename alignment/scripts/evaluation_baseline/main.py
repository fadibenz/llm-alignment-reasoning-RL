from alignment.utils.utils import setup_logging
from pathlib import Path
import json
import logging
from vllm import LLM, SamplingParams
from alignment.scripts.evaluation_baseline.config import parse_args
from alignment.utils.drgrpo_grader import r1_zero_reward_fn
from alignment.inference_utilities.evaluate_vllm import evaluate_vllm

if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    logging.info("Starting evaluation..."
                 f"Using the following arguments: {vars(args)}")

    input_path = Path(args.dataset)
    output_path = Path(args.output_path)
    prompt_path = Path(args.prompt_path)

    if input_path.suffix != "jsonl":
        logging.error("Expects data to be in jsonl/json format")
        raise ValueError

    prompt_template = prompt_path.read_text(encoding="utf-8")

    prompt_list = []
    ground_truth_list = []

    logging.info(f"Loading input file {args.input_file}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json_object = json.loads(line)
                    prompt = prompt_template.replace("{question}", json_object["problem"])
                    prompt_list.append(prompt)
                    ground_truth_list.append(json_object["answer"])
    except Exception as e:
        logging.error(f"Failed reading from input file{e}")
        raise Exception

    logging.info(f"Loading model {args.model_name}")

    vllm_model = LLM(
        model=args.model_name,
        max_num_batched_tokens=args.max_batched_tokens
    )

    eval_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    def make_reward_fn(ground_truths):
        def reward_fn(output_text, idx):
            ref = ground_truths[idx]
            return r1_zero_reward_fn(output_text, ref)
        return reward_fn

    reward_fn = make_reward_fn(ground_truth_list)

    logging.info("Starting evaluation")
    evaluate_vllm(vllm_model, reward_fn, prompt_list, eval_sampling_params, output_path)
    logging.info(f"Finished evaluation, saved results to {output_path}")