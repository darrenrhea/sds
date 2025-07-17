import os
from get_normalization_and_chw_transform import (
     get_normalization_and_chw_transform
)
import numpy as np
from blackpad_preprocessor import blackpad_preprocessor
from load_datapoints_in_parallel import load_datapoints_in_parallel
from typing import Optional, List
from get_training_augmentation import get_training_augmentation
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from WarpDataset import WarpDataset

from colorama import Fore, Style
# from visualize_dataset import visualize_dataset
from typing import Tuple
from torch.utils.data.distributed import DistributedSampler


def get_train_dataloader(
    augmentation_strategy_id: str,
    train_on_binarized_masks: bool,
    datapoint_path_tuples: List[Tuple[Path, Path, Optional[Path]]],
    workers_per_gpu: int,
    frame_height: int,
    frame_width: int,
    patch_width: int,
    patch_height: int,
    batch_size: int,
    do_mixed_precision: bool,
    do_padding: bool,
    train_patches_per_image: int,
):
    """
    Returns a training dataloader especially designed for use with DistributedDataParallel.
    """
    assert isinstance(workers_per_gpu, int), f"workers_per_gpu must be an integer, not {type(workers_per_gpu)}"
    assert workers_per_gpu > 0, f"workers_per_gpu must be positive, not {workers_per_gpu}"

    if frame_width == patch_width and frame_height == patch_height:
        assert train_patches_per_image == 1, "There is no point in training with more than one patch per image if the patch size is the same as the frame size."
   

  
    num_class = 1
    assert num_class == 1
    

    # TODO: somehow determine what normalization to use
    normalization_and_chw_transform = get_normalization_and_chw_transform()
    
    augment = get_training_augmentation(
        augmentation_id=augmentation_strategy_id, 
        frame_width=frame_width, 
        frame_height=frame_height
    )
    
    if do_padding:
        assert frame_height == 1088, "doing padding to force frames to 1920x1088"
        assert frame_width == 1920, "doing padding to force frames to 1920x1088"
        print(f"{Fore.YELLOW}WARNING: doing padding to force frames to 1920x1088{Style.RESET_ALL}")

        # preprocessor = reflect_preprocessor  # fix this
        preprocessor = blackpad_preprocessor

        preprocessor_params = dict(
            desired_height=frame_height,
            desired_width=frame_width
        )
    else:
        preprocessor = None
        preprocessor_params = dict()

    print(f"process id {os.getpid()}")
    channel_stacks = load_datapoints_in_parallel(
        datapoint_path_tuples = datapoint_path_tuples,
        preprocessor = preprocessor,
        preprocessor_params = preprocessor_params,
    )

    for channel_stack in channel_stacks:
        assert channel_stack.shape[0] == frame_height, f"{channel_stack.shape=} {frame_height=}"
        assert channel_stack.shape[1] == frame_width
        assert channel_stack.shape[2] == 5, "aren't we expecting 5 channels since we are using relevance masks?"
        assert channel_stack.dtype == np.uint8
        assert channel_stack.ndim == 3
    
    train_dataset = WarpDataset(
        channel_stacks = channel_stacks,
        train_patches_per_image = train_patches_per_image,
        patch_width = patch_width,
        patch_height = patch_height,
        output_binarized_masks = train_on_binarized_masks,
        normalization_and_chw_transform = normalization_and_chw_transform,
        augment = augment,
        num_mask_channels = num_class
    )

   

    # https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html#distributing-input-data
    # shuffling is accomplished by the DistributedSampler
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        # pin_memory=True,  # TODO: shouldn't this be True?
        shuffle=True,  # leave this False, shuffling is accomplished by the DistributedSampler
    )
   
    assert isinstance(train_dataloader, DataLoader)
    return train_dataloader

