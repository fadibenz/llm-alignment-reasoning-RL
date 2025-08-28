# Qwen 2.5 0.5B evaluation, SFT adapters & training loop

This repository implements a production-grade evaluation and supervised fine-tuning (SFT) pipeline for 
measuring and improving Qwen 2.5 0.5B zero-shot performance on the MATH dataset. 
The document below explains the high-level architecture, where to find the implemented components, 
how the SFT loop is organized, testing and reproducibility notes, and pointers for debugging and extending the code.

---

## High-level summary

This project contains three tightly-coupled subsystems:

1. **Evaluation** — code to format MATH validation examples using the `r1_zero` prompt, generate outputs via a vLLM-backed evaluator, compute `format_reward` and `answer_reward`, and serialize per-example debug traces for analysis.
2. **Adapters / helpers** — deterministic utilities required by both evaluation and training: tokenization helpers, next-token scoring (log-probs), entropy computation, masked normalization, generation logging, and SFT micro-batch logic.
3. **SFT training loop** — a scalable supervised fine-tuning loop that trains a policy model using per-token log-probabilities and response masks produced by the adapters.

All components are tested with the project `tests/` harness and designed to be reproducible across runs.

> **IMPORTANT:**
> All the tests are taken from the public repository of CS336 Assignment 5, 2025. 
> The full project is inspired from this assignment.
---

## Repository layout (reflects the screenshot you provided)

The repository is organized to separate evaluation, SFT helpers, and runnable scripts. Key folders:

```
alignment/
├─ sft_helper_methods/
│  ├─ tokenize_prompt_and_output.py
│  ├─ compute_entropy.py
│  ├─ get_response_log_probs.py
│  ├─ masked_normalize.py
│  ├─ log_generations.py
│  └─ sft_microbatch_train_step.py
├─ scripts/
│  ├─ evaluation_baseline/  # evaluation driver and vLLM wiring
│  ├─ supervised_finetuning/ # hydra-configured training entrypoints
│  │  ├─ config.py
│  │  └─ main.py
│  └─ expert_iteration/
├─ prompts/
├─ inference_utilities/
├─ data/
├─ tests/
└─ README.md
```
---

## SFT loop: complexity and architecture

The SFT loop is intentionally non-trivial to support high-quality supervised fine-tuning at scale. 
The following are the major architecture and engineering decisions implemented:

### Multi-process orchestration (Hydra + torch.distributed)

* The training entry point is configured via **Hydra**. All experiment configuration (model paths, optimizer, scheduler, batch sizing, gradient accumulation steps, reproducibility seeds, and output directories) lives in Hydra configs under `scripts/supervised_finetuning/configs`.
* Multi-GPU training uses PyTorch DDP via `torch.distributed.launch` or `torchrun`. 

### Gradient accumulation and micro-batching

* To support large effective batch sizes on limited GPU memory, the loop splits each optimizer step into `gradient_accumulation_steps` micro-batches.
* `sft_microbatch_train_step` receives per-microbatch log-prob tensors and `response_mask` and computes the masked cross-entropy loss only over response tokens. The function scales the loss by `1 / gradient_accumulation_steps` and calls `loss.backward()`.
* After the configured number of micro-batches, the master process performs `optimizer.step()` and `scheduler.step()` and then `optimizer.zero_grad()`.

### Mixed precision & memory optimizations

* The loop uses **AMP (torch.cuda.amp)** for mixed-precision training; `GradScaler` is used to scale gradients automatically during accumulation.

### Logging and experiment tracking (Weights & Biases)

* Training metrics, micro-batch losses, validation scores, and generation diagnostics are logged to **Weights & Biases (wandb)**. Logs include per-step loss,learning rate, and evaluation metrics.
* Each evaluation during training writes a deterministic JSONL snapshot so runs can be analyzed offline without re-running the model.

### vLLM evaluation hack & model synchronization

When using vLLM as a separate, fast-sampling evaluator (on a different GPU than the policy being trained).
I apply a small set of runtime patches and a programmatic weight-injection step so the vLLM process uses the latest policy checkpoint before each rollout phase. 
The repo includes an **installable vLLM patch package** that addresses vLLM/xformers integration issues [read more](https://github.com/fadibenz/vllm-xlformers-patch); 
however, you still need the runtime monkeypatching below to ensure vLLM can be started inside our Hydra + DDP environment and accept externally-loaded weights.

**Why we patch**

* vLLM's internals assume certain distributed/profiling behaviors that conflict with our single-GPU evaluation process running inside a multi-process training job. The runtime patches avoid a profiling assertion and make `torch.distributed.get_world_size()` return a value compatible with our usage.
* We also need a supported, programmatic pathway to inject the policy model weights into the vLLM model object before each evaluation pass. The patched vLLM build included with the repo exposes `load_weights(...)` on the internal runner; the snippet below uses that API.

**Core patch + initialization code** can be found in [`alignment/inference_utilities/parallel_vllm.py`](./alignment/inference_utilities/parallel_vllm.py)


### Sampling and deterministic seeds

* Sampling parameters (temperature, top\_k, top\_p, max\_new\_tokens) are exposed via Hydra and stored in run metadata for each saved generation. Deterministic seeds are recorded with each JSONL entry to reproduce problematic cases.

### Evaluation & metrics

* The evaluation script formats MATH examples using the `r1_zero` template and generates candidate answers using vLLM. For each generated string a `reward_fn` is applied to produce at least two signals: `format_reward` and `answer_reward`.
* The evaluation stores token-level diagnostics (log-probs and token entropy) when available to aid error analysis.

---

## Testing and reproducibility

* Unit tests covering `compute_entropy`, `get_response_log_probs`, `tokenize_prompt_and_output`, `masked_normalize`, `sft_microbatch_train_step`, and `log_generations` live in `tests/` and are run via the project test runner (e.g., `uv run pytest` or `pytest`).

---

## How to run (developer quick-start)

1. Install dependencies using `uv`.
2. Configure Hydra with your `model_path`, and `data_path`).
3. Launch training with DDP and the desired number of GPUs: `torchrun --nproc_per_node=NUM_GPUS scripts/supervised_finetuning/main.py --config-name=...`
4. Run evaluation (two-GPU recommended): one GPU for the policy model and a second for the vLLM instance used for rollouts. The evaluation driver and `evaluate_vllm` helper are in `scripts/evaluation_baseline/`.

---

## Extension points:

I will add in the upcoming days pipelines that use the same logic and adapters for **expert iteration** and **GRPO**

---