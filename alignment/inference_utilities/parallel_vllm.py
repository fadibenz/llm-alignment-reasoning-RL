from vllm import LLM, SamplingParams
from unittest.mock import patch
import torch
from transformers import PreTrainedModel

def init_vllm(model_id: str,
              device: str,
              torch_dtype: torch.device,
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
            device=device,
            dtype=torch_dtype,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization
        )


import torch.distributed as dist
from torch.nn import Module
from vllm import LLM


def sync_weights_to_vllm(
        policy: Module,
        llm: LLM,
        is_master_process: bool,
        is_vllm_process: bool
):
    if is_master_process:
        # Master process (rank 0) gets the state_dict and broadcasts each tensor.
        print("Master process: Broadcasting policy weights...")
        with torch.no_grad():
            state_dict = policy.state_dict()
            for name, tensor in state_dict.items():
                # Ensure tensor is on the correct device before broadcasting
                tensor = tensor.to(f"cuda:{dist.get_rank()}")
                dist.broadcast(tensor, src=0)
        print("Master process: Weight broadcast complete.")

    elif is_vllm_process:
        # vLLM process receives the broadcasted tensors.
        print("vLLM process: Receiving policy weights...")
        # Get the vLLM model's internal state_dict to use as a template for shapes and dtypes.
        vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        vllm_state_dict = vllm_model.state_dict()

        received_state_dict = {}
        for name, template_tensor in vllm_state_dict.items():
            # Create a tensor on the correct device to receive the broadcast.
            # This tensor will be overwritten by the broadcast operation.
            buffer_tensor = torch.empty_like(template_tensor, device=f"cuda:{dist.get_rank()}")
            dist.broadcast(buffer_tensor, src=0)
            received_state_dict[name] = buffer_tensor

        # Load the newly received weights into the vLLM engine.
        vllm_model.load_weights(received_state_dict.items())
        print("vLLM process: Weights loaded into vLLM engine.")



def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """ Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670. """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())