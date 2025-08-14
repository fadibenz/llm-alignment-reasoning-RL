from vllm import LLM, SamplingParams
from unittest.mock import patch
import torch
from transformers import PreTrainedModel

def init_vllm(model_id: str,
              device: str,
              torch_dtype: torch.dtype,
              gpu_memory_utilization: float = 0.85,
              ) -> LLM:
    # Monkeypatch from TRL: # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch( "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch_dtype,
            device=device,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """ Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670. """
    model = policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy
    state_dict = model.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())