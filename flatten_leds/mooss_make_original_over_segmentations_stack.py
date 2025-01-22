from get_visibility_mask_path import (
     get_visibility_mask_path
)
from typing import List, Tuple
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import numpy as np
from get_flat_original_path import (
     get_flat_original_path
)
from get_flat_mask_path import (
     get_flat_mask_path
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)


def combine_mask_and_original(
    original_rgb_hwc_np_u8: np.array,
    mask_hw_np_u8: np.array,
    color: Tuple[int, int, int],
) -> np.array:
    """
    """

    top_layer_rgba_np_uint8 = np.zeros(
        shape=(
            original_rgb_hwc_np_u8.shape[0],
            original_rgb_hwc_np_u8.shape[1],
            4
        ),
        dtype=np.uint8
    )

    top_layer_rgba_np_uint8[:, :, :3] = original_rgb_hwc_np_u8[:, :, :3]
    top_layer_rgba_np_uint8[:, :, 3] = mask_hw_np_u8

    
    bottom_layer_color_np_uint8 = np.zeros(
        shape=(
            original_rgb_hwc_np_u8.shape[0],
            original_rgb_hwc_np_u8.shape[1],
            3
        ),
        dtype=np.uint8
    ) + color

    composition = feathered_paste_for_images_of_the_same_size(
        bottom_layer_color_np_uint8=bottom_layer_color_np_uint8,
        top_layer_rgba_np_uint8=top_layer_rgba_np_uint8,
    )
    return composition


def mooss_make_original_over_segmentations_stack(
    clip_id: str,
    frame_index: int,
    final_model_ids: List[str],
    board_id: str,
    rip_height: int,
    rip_width: int,
    color: Tuple[int, int, int],
):
    """
    It is easier to evaluate a segmentation model when you can see the original.
    It is easier to stack flattened led boards than the
    whole original image ontop of the augmented image.
    """
    
    num_to_stack = len(final_model_ids) + 1 + 1

    stack_hwc_np_u8 = np.zeros(
        shape=(
            rip_height * num_to_stack,
            rip_width,
            3
        ),
        dtype=np.uint8
    )

    flat_original_file_path = get_flat_original_path(
        clip_id=clip_id,
        frame_index=frame_index,
        rip_width=rip_width,
        rip_height=rip_height,
        board_id=board_id,
    )
    
    flat_original_hwc_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=flat_original_file_path
    )

    # topmost is the original:
    stack_hwc_np_u8[:rip_height, :, :] = flat_original_hwc_np_u8

    visibility_mask_path = get_visibility_mask_path(
        clip_id=clip_id,
        frame_index=frame_index,
        board_id=board_id,
        rip_width=rip_width,
        rip_height=rip_height,
    )

    visibility_mask = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=visibility_mask_path
    )

    suppress = np.sum(flat_original_hwc_np_u8, axis=2) < 3
    suppress = (visibility_mask == 0)

    total = np.zeros(
        shape=(
            rip_height,
            rip_width,
        ),
        dtype=np.float32
    )

    for i, final_model_id in enumerate(final_model_ids):

        flat_mask_path = get_flat_mask_path(
            clip_id=clip_id,
            frame_index=frame_index,
            final_model_id=final_model_id,
            rip_width=rip_width,
            rip_height=rip_height,
            board_id=board_id,
        )

        flat_mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
            abs_file_path=flat_mask_path
        )
        # this lets through the color of the original where it is not visible, which should be essentially black
        flat_mask_hw_np_u8[suppress] = 255

        total += flat_mask_hw_np_u8

        composition = combine_mask_and_original(
            original_rgb_hwc_np_u8=flat_original_hwc_np_u8,
            mask_hw_np_u8=flat_mask_hw_np_u8,
            color=color,
        )
      
        stack_hwc_np_u8[
            rip_height * (i+1):rip_height*(i+2),
            :,
            :3
        ] = composition

    total /= len(final_model_ids)
    total_u8 = np.round(total).clip(0, 255).astype(np.uint8)

    composition = combine_mask_and_original(
        original_rgb_hwc_np_u8=flat_original_hwc_np_u8,
        mask_hw_np_u8=total_u8,
        color=color,
    )

    i = len(final_model_ids)
    
    stack_hwc_np_u8[
        rip_height * (i+1):rip_height*(i+2),
        :,
        :3
    ] = composition
        
    return stack_hwc_np_u8

