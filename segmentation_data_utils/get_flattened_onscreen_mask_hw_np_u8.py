from get_flattened_local_file_path import (
     get_flattened_local_file_path
)
from open_a_grayscale_png_barfing_if_it_is_not_grayscale import (
     open_a_grayscale_png_barfing_if_it_is_not_grayscale
)
import numpy as np


def get_flattened_onscreen_mask_hw_np_u8(
    clip_id: str,
    frame_index: int,
    board_id: str,
    rip_width: int,
    rip_height: int,
) -> np.ndarray:
    """
    Given
    a string clip_id,
    an int frame_index,
    a string board_id,
    an int rip_width, and
    an int rip_height,

    this function will return the flattened onsceeen mask as hw_np_u8,
    regardless of whether or not it is locally staged.

    This also handles, per each computer's idiosyncracies,
    where we choose to stage the "blown-out" original video frames on that computer.
    """
    
    kind = "onscreen"

    original_image_path = get_flattened_local_file_path(
        kind=kind,
        clip_id=clip_id,
        frame_index=frame_index,
        board_id=board_id,
        rip_width=rip_width,
        rip_height=rip_height,
    )

    mask_hw_np_u8 = open_a_grayscale_png_barfing_if_it_is_not_grayscale(
        image_path=original_image_path
    )

    assert mask_hw_np_u8.dtype == np.uint8
    assert mask_hw_np_u8.ndim == 2
    assert mask_hw_np_u8.shape[0] == rip_height
    assert mask_hw_np_u8.shape[1] == rip_width
    return mask_hw_np_u8