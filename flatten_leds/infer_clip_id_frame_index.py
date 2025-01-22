from prii_rgb_and_alpha import (
     prii_rgb_and_alpha
)
from RamInRamOutSegmenter import (
     RamInRamOutSegmenter
)
from prii import (
     prii
)
from typing import Dict, List, Optional
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
import numpy as np


def infer_clip_id_frame_index(
    ram_in_ram_out_segmenter: RamInRamOutSegmenter,
    clip_id: str,
    frame_index: int,
    return_rgba: bool,
    verbose: bool,
) -> Optional[np.array]:
    """
    We tend to use the pair of clip_id and frame_index to identify a video frame.
    This is for when you want to see what the result of a segmentation model on a single frame.
    """

 

    original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
   
    if verbose:
        prii(original_rgb_hwc_np_u8)

    mask = ram_in_ram_out_segmenter.infer(
        frame_rgb=original_rgb_hwc_np_u8
    )

    if verbose:
        prii(mask)
    
        prii_rgb_and_alpha(
            rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            alpha_hw_np_u8=mask
        )

  
    if return_rgba:
        rgba = np.zeros(
            shape=(original_rgb_hwc_np_u8.shape[0], original_rgb_hwc_np_u8.shape[1], 4),
            dtype=np.uint8
        )
        rgba[:, :, :3] = original_rgb_hwc_np_u8
        rgba[:, :, 3] = mask
        return rgba
    else:
        return mask
