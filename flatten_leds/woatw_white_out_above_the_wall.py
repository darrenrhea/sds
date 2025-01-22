from write_rgb_and_alpha_to_png import (
     write_rgb_and_alpha_to_png
)
from typing import Optional
import numpy as np
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image as oaciaascgi
)
from print_red import (
     print_red
)
from pathlib import Path
from gpksafasf_get_primary_keyed_segmentation_annotations_from_a_single_folder import (
     gpksafasf_get_primary_keyed_segmentation_annotations_from_a_single_folder
)
import pprint
from map_points_through_homography import (
     map_points_through_homography
)
from gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame import (
     gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame
)
from get_polygon_points_for_brewcub_023094 import (
     get_polygon_points_for_brewcub_023094
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from acptv_assign_closed_polygon_to_value import (
     acptv_assign_closed_polygon_to_value
)
from prii import (
     prii
)


def woatw_white_out_above_the_wall(
    clip_id: str,
    frame_index: int,
    mask_hw_np_u8: Optional[np.ndarray],
) -> np.ndarray:
    """
    Mathieu needs the area above the walls to be white.
    """
    assert (
        clip_id == "brewcub"
    ), f"only works for brewcub since we don't have an above the wall atlas for anything other clip_id"

    np_nx2_list_of_xys_in_023094 = get_polygon_points_for_brewcub_023094()

    maybe_homography = gthtmtfttf_get_the_homography_that_maps_this_frame_to_this_frame(
        clip_id=clip_id,
        src_frame_index=23094,
        dst_frame_index=frame_index,
    )
    
    if maybe_homography is None:
        print_red(f"Apparently tracking is bad enough that {frame_index} is not possible")
        return None
    else:
        homography = maybe_homography
    
    np_nx2_list_of_xys_in_this_frame = (
        map_points_through_homography(
            homography=homography,
            points=np_nx2_list_of_xys_in_023094,
            a_point_is_a_row=True
        )
    )
    


    value = 255  # whiteify / foregroundify the area above the wall
    
    mutated_mask_hw_np_u8 = mask_hw_np_u8.copy()
    
    acptv_assign_closed_polygon_to_value(
        list_of_xys=np_nx2_list_of_xys_in_this_frame,
        value=value,
        victim_image_hw_and_maybe_c_np=mutated_mask_hw_np_u8
    )

    return mutated_mask_hw_np_u8
    


    

