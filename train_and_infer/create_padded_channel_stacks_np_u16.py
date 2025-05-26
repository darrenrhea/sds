from make_preprocessed_channel_stacks_u16 import (
     make_preprocessed_channel_stacks_u16
)
import numpy as np
from blackpad_preprocessor_u16_to_u16 import (
     blackpad_preprocessor_u16_to_u16
)
from typing import List
from colorama import Fore, Style


def create_padded_channel_stacks_np_u16(
    channel_names: List[str],
    local_file_pathed_annotations: List[dict],
    frame_height: int,
    frame_width: int,
    do_padding: bool,
) -> List[np.ndarray]:
    """
    This is almost unnecessary, it just makes sure that padding by 8 pixels
    is done correctly.

    Makes a Python list of hwc numpy.uint16 arrays,
    each of which is a stack of the channels in the order specified in channel_names.

    This function is used to create channel_stacks for a multiple output model.
    In our first application of this code, to make a multiple output model,
    each "channel_stack" is expected to have these 5 channels:
    r g b floor_not_floor depth_map.
    In the past, this was commonly the choice of channels:
    r g b floor_not_floor relevance_mask 
    """
    if do_padding:
        assert frame_height == 1088, "doing padding to force frames to 1920x1088"
        assert frame_width == 1920, "doing padding to force frames to 1920x1088"
        print(f"{Fore.YELLOW}WARNING: doing padding to force frames to 1920x1088{Style.RESET_ALL}")

        # preprocessor = reflect_preprocessor  # fix this
        preprocessor = blackpad_preprocessor_u16_to_u16

        preprocessor_params = dict(
            desired_height=frame_height,
            desired_width=frame_width
        )
    else:
        preprocessor = None
        preprocessor_params = dict()

    channel_stacks = make_preprocessed_channel_stacks_u16(
        local_file_pathed_annotations=local_file_pathed_annotations,
        preprocessor = preprocessor,
        preprocessor_params = preprocessor_params,
        channel_names=channel_names,
    )

    assert isinstance(channel_stacks, list), "ERROR: channel_stacks is not a Python list!"

    for channel_stack in channel_stacks:
        assert channel_stack.shape[0] == frame_height, f"{channel_stack.shape=} {frame_height=}"
        assert channel_stack.shape[1] == frame_width
        assert channel_stack.shape[2] == len(channel_names)
        assert channel_stack.dtype == np.uint16
        assert channel_stack.ndim == 3
    
    return channel_stacks
