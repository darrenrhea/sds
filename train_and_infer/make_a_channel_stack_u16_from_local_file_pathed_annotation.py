import textwrap
from open_mask_image_as_hw_np_u16_even_though_it_is_u8 import (
     open_mask_image_as_hw_np_u16_even_though_it_is_u8
)
from open_just_the_rgb_part_of_image_as_rgb_hwc_np_u16_even_though_it_is_u8 import (
     open_just_the_rgb_part_of_image_as_rgb_hwc_np_u16_even_though_it_is_u8
)
from load_16bit_grayscale_png_file_as_hw_np_u16 import (
     load_16bit_grayscale_png_file_as_hw_np_u16
)
import numpy as np
from pathlib import Path
from typing import List



def make_a_channel_stack_u16_from_local_file_pathed_annotation(
    local_file_pathed_annotation: dict,
    channel_names: List[str],
) -> List[np.ndarray]:
    """
    Define a "channel stack" as a numpy array of shape (frame_height, frame_width, how_many_channels).
    It is a stack of channels, some of which are:
    * inputs to the neural network (like RGB's r, g, b channels, or luma y, u/Cb, v/Cr channels, possibly other models outputs).
    * labels/targets, i.e. the desired outputs of the neural network, floor_not_floor, depth_map, wood_not_wood, etc.
    * Other things that affect how the loss function is calculated, like relevance weights, i.e. how much we care about each pixel.
    * similar, weird things, like the visibility_mask / onscreen_mask,
      that are kind-of an input because we could know them ahead-of-time,
      but also something that affects the loss function (begging it to segment things that are off screen seems impossible, so no penalty).

    Given a local_file_pathed_annotation, which has local_file_paths to all the data about that video frame.
    """
    assert isinstance(local_file_pathed_annotation, dict)
    assert isinstance(channel_names, list), f"ERROR: {channel_names=} is not a list"
    for s in channel_names:
        assert isinstance(s, str), "you gave me a non-string in channel_names: {s}"
    
    possible_mask_u8_names = ["floor_not_floor", "relevance", "visibility"]
    masks_we_are_willing_to_impute = ["relevance"]

    local_file_paths = local_file_pathed_annotation["local_file_paths"]
    assert isinstance(local_file_paths, dict)

    # We will get each desired channel out of local_file_pathed_annotation:
    channels = dict()

    if "r" in channel_names or "g" in channel_names or "b" in channel_names:
        original_frame_path = local_file_paths["original"]
        assert isinstance(original_frame_path, Path)
        assert original_frame_path.is_file(), "ERROR: original_frame_path does not exist yet you are asking for some of the RGB channels"
         # The RGB part to u16 ranging over [0, 65535]
        rgb_hwc_np_u16 = open_just_the_rgb_part_of_image_as_rgb_hwc_np_u16_even_though_it_is_u8(
            image_path=original_frame_path
        )
        assert rgb_hwc_np_u16.dtype == np.uint16
        for i, channel_name in enumerate(["r", "g", "b"]):
            if channel_name in channel_names:
                channels[channel_name] = rgb_hwc_np_u16[:, :, i]

   
    # even if a channel_name is mentioned multiple times, this will only load it once:
    mask_u8_names = [name for name in possible_mask_u8_names if name in channel_names]
    for mask_u8_name in mask_u8_names:
        mask_u8_path = local_file_paths[mask_u8_name]
        if mask_u8_path is None:
            if mask_u8_name in masks_we_are_willing_to_impute:
                mask_hw_np_u16 = np.ones((1080, 1920), dtype=np.uint16) * 65535  # impute as solid white
            else:
                assert (
                    False
                ), textwrap.dedent(
                    f"""
                    ERROR: For
                    {local_file_pathed_annotation}
                    local_file_paths['{mask_u8_name}'] does not exist yet you are asking for the {mask_u8_name} channel
                    """
                )
        else:
            assert isinstance(mask_u8_path, Path)
            assert mask_u8_path.is_file(), f"ERROR: {mask_u8_path} does not exist yet you are asking for the {mask_u8_name} channel"  
            mask_hw_np_u16 = open_mask_image_as_hw_np_u16_even_though_it_is_u8(mask_path=mask_u8_path)
        assert mask_hw_np_u16.dtype == np.uint16
        assert mask_hw_np_u16.ndim == 2
        channels[mask_u8_name] = mask_hw_np_u16

    possible_mask_u16_names = ["depth_map"]
    mask_u16_names = [name for name in possible_mask_u16_names if name in channel_names]
    
    for mask_u16_name in mask_u16_names:
        mask_u16_path = local_file_paths[mask_u16_name]
        assert isinstance(mask_u16_path, Path)
        assert mask_u16_path.is_file(), f"ERROR: {mask_u16_path} does not exist yet you are asking for the {mask_u16_name} channel"
        
        mask_hw_np_u16 = load_16bit_grayscale_png_file_as_hw_np_u16(
            mask_u16_path
        )
        assert mask_hw_np_u16.dtype == np.uint16
        assert mask_hw_np_u16.ndim == 2

        channels[mask_u16_name] = mask_hw_np_u16
   
    stack_list = [
        channels[channel_name]
        for channel_name in channel_names
    ]
    for c in stack_list:
        assert c.dtype == np.uint16
        assert c.ndim == 2
        assert c.shape[0] == 1080
        assert c.shape[1] == 1920
    # raw here means not yet preprocessed:
    channel_stack = np.stack(
        stack_list,
        axis=2
    )
    assert isinstance(channel_stack, np.ndarray)
    assert channel_stack.dtype == np.uint16
    assert (
        channel_stack.shape[2] == len(channel_names)
    ), f"ERROR: {channel_stack.shape[2]=} yet {len(channel_names)=} because {channel_names=}"
    
    return channel_stack
    
