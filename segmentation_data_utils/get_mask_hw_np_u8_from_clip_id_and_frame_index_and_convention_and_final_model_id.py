from get_human_annotated_mask_path_from_clip_id_and_frame_index_and_segmentation_convention import (
     get_human_annotated_mask_path_from_clip_id_and_frame_index_and_segmentation_convention
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from pathlib import Path
from typing import Optional
import numpy as np




def get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id(
    clip_id: str,
    frame_index: int,
    segmentation_convention: str,
    final_model_id: str,
) -> Optional[np.ndarray]:
    """
    Eventually, we want this to return that mask at that frame
    regardless of whether or not it is locally staged
    and regardless of whether or not it has already been calculated or not.

    Returns hw_np_u8 or None if the mask is not available.
    """

   

    if final_model_id == "human":

        mask_path = get_human_annotated_mask_path_from_clip_id_and_frame_index_and_segmentation_convention(
            clip_id=clip_id,
            frame_index=frame_index,
            segmentation_convention=segmentation_convention,
        )
    else:
         raise Exception("is not implemented")

    # if you have gotten this far, you have a mask_path
    assert isinstance(mask_path, Path)
    assert mask_path.is_file(), f"{mask_path=} is not a file"

    mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=mask_path
    )

    assert mask_hw_np_u8.dtype == np.uint8
    assert mask_hw_np_u8.ndim == 2
    assert mask_hw_np_u8.shape[0] == 1080
    assert mask_hw_np_u8.shape[1] == 1920

    return mask_hw_np_u8