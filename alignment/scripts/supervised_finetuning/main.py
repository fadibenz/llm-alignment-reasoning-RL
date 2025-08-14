import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
from rich.pretty import pprint as pprint
from rich.traceback import install
import wandb
from vllm import SamplingParams

from alignment.inference_utilities.evaluate_and_metrics import evaluate_and_metrics
from alignment.inference_utilities.parallel_vllm import init_vllm, load_policy_into_vllm_instance
from alignment.inference_utilities.reward_function import make_reward_fn
from alignment.sft_helper_methods.log_generations import log_generations

from alignment.utils.dist_utils import _setup_process_group, _cleanup_process_group
from alignment.sft_helper_methods.get_response_log_probs import get_response_log_probs
from alignment.scripts.supervised_finetuning.config import register_configs, Config
from alignment.sft_helper_methods.sft_microbatch_train_step import sft_microbatch_train_step
from alignment.sft_helper_methods.tokenize_prompt_and_output import tokenize_prompt_and_output
from alignment.utils.optimizer import get_cosine_lr
from alignment.utils.utils import setup_logging, set_seed_everything
from alignment.utils.data import load_validation_data, load_training_data, TokenizedDataset

logger = logging.getLogger(__name__)
register_configs()

install(show_locals=True)

@hydra.main(version_base=None, config_path=str(Path("configs").absolute().resolve()), config_name="config")
def main(cfg: Config) -> None:
    setup_logging()
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    default_cfg = OmegaConf.structured(Config())
    cfg = OmegaConf.merge(default_cfg, cfg_dict)
    pprint(cfg)

    # Setting up distributed training
    device, rank, world_size, local_rank = _setup_process_group()

    if world_size < 2:
        raise RuntimeError("world_size must be >=2 (one for training, one reserved GPU for vLLM). "
                           "Or remove the reserved GPU logic and run with world_size==num_training_gpus.")

    seed = cfg.training.seed + rank
    set_seed_everything(seed)

    training_processes_list = list(range(world_size - 1))
    vllm_rank =  world_size - 1

    is_ddp = (world_size > 2)

    is_master_process = (rank == 0)
    is_training_process = rank in training_processes_list
    training_group = dist.new_group(ranks=training_processes_list)

    # Initialize vllm on DIFFERENT DEVICE
    # This is a neat trick used in TRL (HF) to use GPU 0 as a
    # "remote control" for inference on another GPU.

    torch_dtype = getattr(torch, cfg.training.dtype)

    if is_master_process:
        vllm_local_rank = vllm_rank % world_size
        vllm_device = f"cuda:{vllm_local_rank}"
        vllm_model = init_vllm(cfg.model.model_name, vllm_device, torch_dtype, cfg.inference.gpu_utilization)
    else:
        vllm_model = None

    eval_sampling_params = SamplingParams(
        temperature=cfg.inference.temperature,
        top_p=cfg.inference.top_p,
        logprobs=cfg.inference.logprobs,
        stop=["</answer>"],
        include_stop_str_in_output=cfg.inference.include_stop_str_in_output,
        seed=seed
    )


    if is_master_process:
        # Load validation data
        prompt_template = Path(cfg.paths.prompt_path).read_text(encoding="utf-8")
        prompts, ground_truths = load_validation_data(cfg.paths.valid_path, prompt_template)
        reward_fn = make_reward_fn(ground_truths)

        # Setup wandb
        if cfg.training.wandb_project and cfg.training.wandb_entity:
            wandb.init(
                entity=cfg.training.wandb_entity,
                project=cfg.training.wandb_project,
                config=OmegaConf.to_container(cfg, resolve=True),
                name=cfg.paths.model_output.name,
            )

            wandb.define_metric("train_step")
            wandb.define_metric("eval_step")
            wandb.define_metric("train/*", step_metric="train_step")
            wandb.define_metric("eval/*", step_metric="eval_step")

        # Create eval directory
        eval_output_dir = cfg.paths.model_output / "evaluation_results"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

    else:
        prompts, ground_truths, reward_fn, eval_output_dir = None, None, None, None

    # Start training
    if is_training_process:
        prompt_strs, output_strs = load_training_data(cfg.paths.train_path, cfg.training.number_train_samples)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        data_dict = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

        dataset = TokenizedDataset(
            data_dict["input_ids"],
            data_dict["labels"],
            data_dict["response_mask"]
        )

        if is_ddp:
            sampler = DistributedSampler(
                dataset,
                num_replicas=len(training_processes_list),
                rank=rank,
                shuffle=True,
                seed=seed
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            dataset,
            batch_size=cfg.training.train_batch_size,
            shuffle=shuffle,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            persistent_workers=(cfg.training.num_workers > 0),
            sampler = sampler
        )

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.model.model_name,
            # load model in FP32 for scaler
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )

        if is_master_process:
            pprint(model)

        if cfg.training.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Move Policy to correct GPU
        model = model.to(device)

        if cfg.training.use_compile:
            model = torch.compile(model)

        if is_ddp:
            model = DDP(
                model,
                device_ids=[local_rank],
                process_group=training_group
            )

        # Setup Optimizer
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
        params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": params_to_decay, "weight_decay": cfg.training.weight_decay},
            {"params": params_to_not_decay, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=cfg.training.lr,
            betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
            fused=True,
        )
        # Mixed-precision training
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
        scaler = torch.amp.GradScaler(enabled=(torch_dtype == torch.float16))

        train_step_counter = 0
        eval_step_counter = 0
        optimizer_step = 0 # "effective step"
        train_steps = cfg.training.num_epochs * (len(train_loader) // cfg.training.gradient_accumulation_steps)

        for epoch in range(cfg.training.num_epochs):
                pbar = tqdm(
                    train_loader,
                    desc=f"Training, Epoch {epoch+ 1}/{cfg.training.num_epochs}",
                    unit="batch",
                    disable= not is_master_process
                )

                if is_ddp:
                    sampler.set_epoch(epoch)

                for idx, (batch_ids, batch_labels, batch_masks) in enumerate(pbar):
                    model.train()
                    if is_ddp:
                        model.require_backward_grad_sync = ((idx + 1) % cfg.training.gradient_accumulation_steps == 0)

                    batch_ids = batch_ids.to(device, non_blocking=True)
                    batch_labels = batch_labels.to(device, non_blocking=True)
                    batch_masks = batch_masks.to(device, non_blocking=True)

                    with amp_ctx:
                        response_probs = get_response_log_probs(
                            model,
                            batch_ids,
                            batch_labels,
                            False
                        )

                        loss, metadata = sft_microbatch_train_step(
                            response_probs["log_probs"],
                            batch_masks,
                            cfg.training.gradient_accumulation_steps,
                        )

                    scaler.scale(loss).backward()

                    if (idx + 1) % cfg.training.gradient_accumulation_steps == 0:

                        lr = get_cosine_lr(
                            optimizer_step,
                            max_learning_rate=cfg.training.lr,
                            min_learning_rate=cfg.training.lr * 0.1,
                            warmup_iters=int(train_steps * cfg.training.warmup_ratio),
                            cosine_cycle_iters=train_steps,
                        )

                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr

                        scaler.unscale_(optimizer)
                        if cfg.training.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        loss_float = loss.item() * cfg.training.gradient_accumulation_steps

                        if is_master_process:
                            pbar.set_description(f"Loss: {loss_float:.4f}")

                            if cfg.training.wandb_project and ((optimizer_step + 1) % cfg.training.log_interval == 0):
                                wandb.log({
                                    "train/loss": loss_float,
                                    "train/lr": lr,
                                    "train_step": train_step_counter
                                })

                                train_step_counter += 1

                            if (optimizer_step + 1) % cfg.training.eval_interval == 0 :
                                load_policy_into_vllm_instance(model, vllm_model)
                                metrics = evaluate_and_metrics(vllm_model, reward_fn, prompts, ground_truths, eval_sampling_params, cfg.inference.sample_size)

                                wandb.log({
                                    "eval/answer_reward_mean": metrics["answer_reward_mean"],
                                    "eval/format_reward_mean": metrics["format_reward_mean"],

                                    "eval/avg_response_length": metrics["avg_response_length"],
                                    "eval/avg_length_correct": metrics["avg_length_correct"],
                                    "eval/avg_length_incorrect": metrics["avg_length_incorrect"],

                                    "eval/avg_token_entropy": metrics["avg_token_entropy"],
                                    "eval/avg_entropy_correct": metrics["avg_entropy_correct"],
                                    "eval/avg_entropy_incorrect": metrics["avg_entropy_incorrect"],

                                    "eval_step": eval_step_counter
                                })
                                eval_step_counter += 1

                                log_generations(
                                    samples=metrics.get('random_samples', []),
                                    logger=logger
                                )

                                if cfg.training.save_checkpoints:
                                    model_weights_output_path = cfg.paths.model_output / f"step_{optimizer_step:010d}" / "model.pt"
                                    model_weights_output_path.parent.mkdir(parents=True, exist_ok=True)
                                    to_save = model.module if isinstance(model, DDP) else model
                                    torch.save(to_save.state_dict(), model_weights_output_path)

                        optimizer_step += 1

    _cleanup_process_group()

if __name__ == "__main__":
    main()
