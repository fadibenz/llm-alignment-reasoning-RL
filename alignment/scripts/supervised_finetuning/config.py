from pathlib import Path
from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
from hydra.core.config_store import ConfigStore

@dataclass
class PathsConfig:
    train_path: Path = MISSING
    valid_path: Path = MISSING
    prompt_path: Path = MISSING
    model_output: Path = MISSING

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"

@dataclass
class TrainingConfig:

    number_train_samples: int = 256
    num_epochs: int = 5
    train_batch_size: int = 128

    seed: int = 2025
    dtype: str = "float16"
    eval_batch_size: int = "${training.train_batch_size}"
    eval_interval: int = 20
    gradient_accumulation_steps: int = 2
    max_grad_norm: float | None = 1.0
    num_workers: int = 4
    lr: float = 1e-3
    flash_attention: str = "flash_attention_2"
    warmup_ratio: float = 0.01
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    wandb_project: str | None = None
    wandb_entity: str | None = None
    log_interval: int = 20
    save_checkpoints: bool = False

@dataclass
class InferenceConfig:
    temperature: int = 1
    top_p: int = 1
    logprobs: int = 10
    stop_sequences: str = "</answer>"
    include_stop_str_in_output: bool = True
    sample_size: int = 5

@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig= field(default=InferenceConfig)


def register_configs():
    OmegaConf.register_resolver("eval", eval)
    cs = ConfigStore.instance()
    cs.store(group="inference", name="base_inference", node=InferenceConfig)
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="model", name="base_model", node=ModelConfig)
    cs.store(group="paths", name="base_paths", node=PathsConfig)

    cs.store(name="base_config", node=Config)