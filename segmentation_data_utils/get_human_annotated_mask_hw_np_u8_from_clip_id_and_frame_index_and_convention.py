from print_red import (
     print_red
)
from get_human_annotated_mask_path_from_clip_id_and_frame_index_and_segmentation_convention import (
     get_human_annotated_mask_path_from_clip_id_and_frame_index_and_convention
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from pathlib import Path
from typing import Optional
import numpy as np

 
def get_human_annotated_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention(
    clip_id: str,
    frame_index: int,
    segmentation_convention: str,
) -> Optional[np.ndarray]:

    
    mask_path = get_human_annotated_mask_path_from_clip_id_and_frame_index_and_convention(
        clip_id=clip_id,
        frame_index=frame_index,
        segmentation_convention=segmentation_convention,
    )
    
    # if you have gotten this far, you have a mask_path
    assert isinstance(mask_path, Path)
    if not mask_path.is_file():
        print_red(f"{mask_path=} is not a file")
        return None

    mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=mask_path
    )

    assert mask_hw_np_u8.dtype == np.uint8
    assert mask_hw_np_u8.ndim == 2
    assert mask_hw_np_u8.shape[0] == 1080
    assert mask_hw_np_u8.shape[1] == 1920

    return mask_hw_np_u8

