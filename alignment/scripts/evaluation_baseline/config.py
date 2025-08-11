import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Script to evaluate model on a dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Name of HF model to evaluate")
    parser.add_argument("--dataset", type=str, default="data/MATH/validation.jsonl",
                        help="Path to dataset.jsonl to evaluate on")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save serialized evaluation results")
    parser.add_argument("--prompt_path", type=str, default="alignment/prompts/r1_zero.prompt",
                        help="Path to prompt to use")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite the file at output_path")

    # vLLM params
    parser.add_argument("--max_batched_tokens", type=int, default=8192,
                        help="Maximum number of batched tokens for vllm")

    parser.add_argument("--stop_sequences", nargs="+", default=["</answer>"],
                        help="Stop sequence for vllm")
    parser.add_argument("--include_stop_str_in_output", type=bool, default=True,
                        help="Whether to include the stop str in output")

    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Sampling top-p")

    return parser.parse_args()