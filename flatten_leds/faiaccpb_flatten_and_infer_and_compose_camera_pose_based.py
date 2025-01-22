from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask import (
     uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask
)
from fasf_flatten_a_single_frame import (
     fasf_flatten_a_single_frame
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
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


def faiaccpb_flatten_and_infer_and_compose_camera_pose_based(
    ram_in_ram_out_segmenter: RamInRamOutSegmenter,
    clip_id: str,
    frame_index: int,
    board_ids: List[str],
    board_id_to_rip_height: Dict[str, int],
    board_id_rip_width: Dict[str, int],
    # board_id_to_new_ad: Dict[str, np.array],
    return_rgba: bool,
    verbose: bool,
) -> Optional[np.array]:
    """
    TODO: priseg and priflatseg are both calling this,
    which could be a problem.

    WARNING: This basketball-only, because it is camera-pose-based, until baseball migrate to camera-poses.

    This is for when you want to see what the total result of

    1. flattening
    2. segmenting each of the flattened ad boards
    3. then compositing
    
    looks like for a single frame.

    This procedure,
    faiac_flatten_and_infer_and_compose,

    is kept to actually doing composition, such as for making demo videos.
    There is another procedure, masfvp_make_a_single_flattened_vip_preannotation,
    which is kept for making preannotations which have the special issue of needing to over-do the margins,
    BUT IT IS ONLY FOR BASEBALL SINCE IT IS HOMOGRAPHY BASED.
    """

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0
    )

    rip_height = 256
    rip_width = 4268

    original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    photograph_height_in_pixels = original_rgb_hwc_np_u8.shape[0]
    photograph_width_in_pixels = original_rgb_hwc_np_u8.shape[1]

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    
    if verbose:
        prii(original_rgb_hwc_np_u8)

    board_id_to_pair_of_rgb_and_visibility_mask = {}
    
    for board_id in board_ids:
        rip_height = board_id_to_rip_height[board_id]
        rip_width = board_id_rip_width[board_id]

        (
            rgb_rip,
            onscreen_mask
        ) = fasf_flatten_a_single_frame(
            clip_id=clip_id,
            frame_index=frame_index,
            rip_height=rip_height,
            rip_width=rip_width,
            board_id=board_id,
        )

        if verbose:
            print(f"{clip_id=} {frame_index=} {board_id=}")
            prii(rgb_rip)
            prii(onscreen_mask)
       
        mask = ram_in_ram_out_segmenter.infer(
            frame_rgb=rgb_rip
        )
        cleaned_mask = np.minimum(mask, onscreen_mask)
        if verbose:
            print(f"the mask for {board_id=}:")
            prii(cleaned_mask)
        
            prii_rgb_and_alpha(
                rgb_hwc_np_u8=rgb_rip,
                alpha_hw_np_u8=cleaned_mask
            )

        board_id_to_pair_of_rgb_and_visibility_mask[board_id] = {
            "rgb": rgb_rip,
            "visibility_mask": onscreen_mask,
            "cleaned_mask": cleaned_mask,
        }
    
    assert len(ad_placement_descriptors) == 1

    pairs_of_flat_mask_and_ad_descriptor = [
        (
            board_id_to_pair_of_rgb_and_visibility_mask["board0"]["cleaned_mask"],
            ad_placement_descriptors[0]
        ),
    ]
    
    total_mask = uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask(
        pairs_of_flat_mask_and_ad_descriptor=pairs_of_flat_mask_and_ad_descriptor,
        camera_pose=camera_pose,
        photograph_height_in_pixels=photograph_height_in_pixels,
        photograph_width_in_pixels=photograph_width_in_pixels,
    )

    if return_rgba:
        rgba = np.zeros(
            shape=(original_rgb_hwc_np_u8.shape[0], original_rgb_hwc_np_u8.shape[1], 4),
            dtype=np.uint8
        )
        rgba[:, :, :3] = original_rgb_hwc_np_u8
        rgba[:, :, 3] = total_mask
        return rgba

    if verbose:
        prii(total_mask)
    # background color, solid green
    bottom_layer_color_np_uint8 = np.zeros_like(original_rgb_hwc_np_u8) + [0, 80, 0]

    top_layer_rgba_np_uint8 = np.zeros(
        shape=(original_rgb_hwc_np_u8.shape[0], original_rgb_hwc_np_u8.shape[1], 4),
        dtype=np.uint8
    )
    top_layer_rgba_np_uint8[:, :, :3] = original_rgb_hwc_np_u8
    top_layer_rgba_np_uint8[:, :, 3] = total_mask

    ans = feathered_paste_for_images_of_the_same_size(
        bottom_layer_color_np_uint8=bottom_layer_color_np_uint8,
        top_layer_rgba_np_uint8=top_layer_rgba_np_uint8
    )
    
    return ans
