from pathlib import Path
from typing import List, Optional, Tuple
from ddp_setup import (
     ddp_setup
)
from get_all_annotations_in_these_folders import (
     get_all_annotations_in_these_folders
)
from get_train_dataloader_for_ddp import (
     get_train_dataloader_for_ddp
)
import torch.multiprocessing as mp
import torch
from colorama import Fore, Style


def get_datapoint_path_tuples_for_testing(
) -> List[Tuple[Path, Path, Optional[Path]]]:
    dataset_folder_strs = [
        "/shared/flattened_fake_game5/state_farm_state_farm",
    ]

    dataset_folders = [
        Path(dataset_folder_str).expanduser().resolve()
        for dataset_folder_str in dataset_folder_strs
    ]

    datapoint_path_tuples = get_all_annotations_in_these_folders(
        dataset_folders=dataset_folders,
        diminish_to_this_many=10000,
    )

    num_training_points = len(datapoint_path_tuples)
    print(f"{Fore.YELLOW}{num_training_points=}{Style.RESET_ALL}")
    return datapoint_path_tuples


def main(
    rank: int,
    world_size: int,
    dict_of_arguments: dict,
):
    world_size = dict_of_arguments["world_size"]
    ddp_setup(rank, world_size)

    datapoint_path_tuples = get_datapoint_path_tuples_for_testing()

    augmentation_strategy_id = "forflat"
    train_on_binarized_masks = False
    workers_per_gpu = 4
    frame_height = 256
    frame_width = 1856
    patch_width = 1856
    patch_height = 256
    batch_size = 8
    do_mixed_precision = True
    do_padding = False
    train_patches_per_image = 1
    do_mixed_precision = True
    do_padding = False
    train_patches_per_image = 1

    train_dataloader = get_train_dataloader_for_ddp(
        augmentation_strategy_id=augmentation_strategy_id,
        train_on_binarized_masks=train_on_binarized_masks,
        datapoint_path_tuples=datapoint_path_tuples,
        workers_per_gpu=workers_per_gpu,
        frame_height=frame_height,
        frame_width=frame_width,
        patch_width=patch_width,
        patch_height=patch_height,
        batch_size=batch_size,
        do_mixed_precision=do_mixed_precision,
        do_padding=do_padding,
        train_patches_per_image=train_patches_per_image
    )
    

    # Prove that the dataloaders are iterables such that if you
    # turn them into an iterator (for instance, by for looping over them)
    # with each iteration you get one torch.Tensor of shape
    # batch_size x 5 channels x 1088 height x 1920 width
    thing = next(iter(train_dataloader))
    assert isinstance(thing, torch.Tensor)
    assert thing.device.type == 'cpu'
    assert thing.shape == (batch_size, 5, patch_height, patch_width)
    print(f"{thing.shape=}")


def test_get_train_dataloader_for_ddp_1():
    
    # two processes
    world_size = 2

    dict_of_arguments = {
        "world_size": world_size
    }
    mp.spawn(  # Spawns nprocs processes that run fn with an integer called rank plus args.
        fn=main,  # run this process
        args=(world_size, dict_of_arguments, ), # pass in the rank and these other arguments to it in that order
        nprocs=world_size  # this many copies of the process
    )



if __name__ == "__main__":
    test_get_train_dataloader_for_ddp_1()
