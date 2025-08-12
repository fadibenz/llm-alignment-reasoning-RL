import os

import torch
import torch.distributed as dist


def _setup_process_group():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    device_count = torch.cuda.device_count()
    if device_count > 0:
        local_rank = rank % device_count
        torch.cuda.set_device(local_rank)
    else:
        raise ValueError("Unable to find CUDA devices.")
    device = f"cuda:{local_rank}"

    # initialize the process group
    dist.init_process_group("ncll", rank=rank, world_size=world_size)
    return device, local_rank, world_size


def _cleanup_process_group():
    # Synchronize before we destroy the process group
    dist.barrier()
    dist.destroy_process_group()
