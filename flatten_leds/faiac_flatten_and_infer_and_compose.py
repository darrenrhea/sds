from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from pathlib import Path
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from amthr_add_margins_to_homographic_rectangle import (
     amthr_add_margins_to_homographic_rectangle
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from umbihr_unflatten_mask_back_into_homographic_rectangle import (
     umbihr_unflatten_mask_back_into_homographic_rectangle
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
from get_screen_corners_from_clip_id_and_frame_index import (
     get_screen_corners_from_clip_id_and_frame_index
)
from fhrwvm_flatten_homographic_rectangle_with_visibility_mask import (
     fhrwvm_flatten_homographic_rectangle_with_visibility_mask
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
import numpy as np


def faiac_flatten_and_infer_and_compose(
    ram_in_ram_out_segmenter: RamInRamOutSegmenter,
    clip_id: str,
    frame_index: int,
    board_ids: List[str],
    board_id_to_rip_height: Dict[str, int],
    board_id_rip_width: Dict[str, int],
    # board_id_to_new_ad: Dict[str, np.array],
    verbose: bool = False
) -> Optional[np.array]:
    """
    WARNING: baseball / homography based only.

    This is for when you want to see what the total result of

    1. flattening
    2. segmenting each of the flattened ad boards
    3. then compositing
    
    looks like for a single frame.

    This procedure,
    faiac_flatten_and_infer_and_compose,

    is kept to actually doing composition, such as for making demo videos.
    There is another procedure, masfvp_make_a_single_flattened_vip_preannotation,
    which is kept for making preannotations which have the special issue of needing to over-do the margins.
    """
    
    screen_name_to_corner_name_to_xy = (
        get_screen_corners_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )
    )
    if "left" not in screen_name_to_corner_name_to_xy or "right" not in screen_name_to_corner_name_to_xy:
        print(f"WARNING: not all boards are present for {clip_id=}, {frame_index=}, returning None")
        return None
    
    if verbose:
        print(screen_name_to_corner_name_to_xy)

    image = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )
    
    if verbose:
        print(f"{clip_id=} {frame_index=}")
        prii(image)

    present_board_ids = sorted(
        [
            x
            for x in board_ids
            if x in screen_name_to_corner_name_to_xy
        ]
    )
                
    if present_board_ids != board_ids:
        print(f"WARNING: not all board_ids are present for {clip_id=}, {frame_index=}")
    
    board_id_to_pair_of_rgb_and_visibility_mask = {}
    
    for board_id in board_ids:
        rip_height = board_id_to_rip_height[board_id]
        rip_width = board_id_rip_width[board_id]

        corner_name_to_xy = screen_name_to_corner_name_to_xy[board_id]
        
        tl_bl_br_tr = [corner_name_to_xy[corner] for corner in ["tl", "bl", "br", "tr"]]

        interior_homographic_rectangle = np.array(
            tl_bl_br_tr,
            dtype=np.float32
        )

        full_margin_added_homographic_rectangle = amthr_add_margins_to_homographic_rectangle(
            interior_homographic_rectangle=interior_homographic_rectangle,
            out_height=rip_height,
            pad_frac_height=0.15,
            out_width=rip_width,
        )

        # over_margin_added_homographic_rectangle = amthr_add_margins_to_homographic_rectangle(
        #     interior_homographic_rectangle=interior_homographic_rectangle,
        #     out_height=rip_height,
        #     pad_frac_height=0.25,
        #     out_width=rip_width,
        # )

        # half_margin_added_homographic_rectangle = amthr_add_margins_to_homographic_rectangle(
        #     interior_homographic_rectangle=interior_homographic_rectangle,
        #     out_height=rip_height,
        #     pad_frac_height=0.15/2,
        #     out_width=rip_width,
        # )

        rgb_rip, onscreen_mask = fhrwvm_flatten_homographic_rectangle_with_visibility_mask(
            image=image,
            tl_bl_br_tr_np=full_margin_added_homographic_rectangle,
            out_height=rip_height,
            out_width=rip_width,
        )
 
        print(f"{clip_id=} {frame_index=} {board_id=}")
    
        # print(f"WTF {board_id}=")
        # prii(rgb_rip)
        # prii(onscreen_mask)

        
        mask = ram_in_ram_out_segmenter.infer(
            frame_rgb=rgb_rip
        )
        cleaned_mask = np.minimum(mask, onscreen_mask)
        # prii(cleaned_mask)
        
        # prii_rgb_and_alpha(
        #     rgb_hwc_np_u8=rgb_rip,
        #     alpha_hw_np_u8=cleaned_mask
        # )

        board_id_to_pair_of_rgb_and_visibility_mask[board_id] = {
            "rgb": rgb_rip,
            "visibility_mask": onscreen_mask,
            "cleaned_mask": cleaned_mask,
            "full_margin_added_homographic_rectangle": full_margin_added_homographic_rectangle,
            # "half_margin_added_homographic_rectangle": half_margin_added_homographic_rectangle,
            # "over_margin_added_homographic_rectangle": over_margin_added_homographic_rectangle,
            "interior_homographic_rectangle": interior_homographic_rectangle,
        }

    total_mask1 = 255 * np.ones_like(image[:, :, 0])
    for board_id in board_ids:
        flattened_mask = board_id_to_pair_of_rgb_and_visibility_mask[board_id]["cleaned_mask"]
        board_as_4_corners_within_full_image = board_id_to_pair_of_rgb_and_visibility_mask[board_id]["full_margin_added_homographic_rectangle"]
       
        full_mask = umbihr_unflatten_mask_back_into_homographic_rectangle(
            flattened_mask=flattened_mask,  # Usually 1024 x 256
            board_as_4_corners_within_full_image=board_as_4_corners_within_full_image,
            full_height=1080,
            full_width=1920,
        )
        total_mask1 = np.minimum(total_mask1, full_mask)
    
    # prii(total_mask1, caption="total_mask1:")

    draw_limiter = 255 * np.ones_like(image[:, :, 0])
    for board_id in board_ids:
        flattened_mask = np.zeros(shape=(256, 1024), dtype=np.uint8)
        
        board_as_4_corners_within_full_image = (
            # board_id_to_pair_of_rgb_and_visibility_mask[board_id]["interior_homographic_rectangle"]
            # board_id_to_pair_of_rgb_and_visibility_mask[board_id]["half_margin_added_homographic_rectangle"]
            board_id_to_pair_of_rgb_and_visibility_mask[board_id]["full_margin_added_homographic_rectangle"]
            # board_id_to_pair_of_rgb_and_visibility_mask[board_id]["over_margin_added_homographic_rectangle"]
        )

        full_mask = umbihr_unflatten_mask_back_into_homographic_rectangle(
            flattened_mask=flattened_mask,  # Usually 1024 x 256
            board_as_4_corners_within_full_image=board_as_4_corners_within_full_image,
            full_height=1080,
            full_width=1920,
        )
        draw_limiter = np.minimum(draw_limiter, full_mask)
    

    # prii(draw_limiter, caption="draw_limiter:")

    total_mask = np.maximum(total_mask1, draw_limiter)
    # prii(total_mask)

    # solid red
    bottom_layer_color_np_uint8 = np.zeros_like(image) + [80, 0, 0]

    top_layer_rgba_np_uint8 = np.zeros(
        shape=(image.shape[0], image.shape[1], 4),
        dtype=np.uint8
    )
    top_layer_rgba_np_uint8[:, :, :3] = image
    top_layer_rgba_np_uint8[:, :, 3] = total_mask

    ans = feathered_paste_for_images_of_the_same_size(
        bottom_layer_color_np_uint8=bottom_layer_color_np_uint8,
        top_layer_rgba_np_uint8=top_layer_rgba_np_uint8
    )
    
    return ans
