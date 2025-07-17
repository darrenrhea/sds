import torch
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup(
    rank: int,
    world_size: int
):
    """
    Each process should call this at its beginning.
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
        https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
