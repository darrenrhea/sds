from get_click_on_image_by_two_stage_zoom import (
     get_click_on_image_by_two_stage_zoom
)
from make_rgb_hwc_np_u8_from_rgba_hwc_np_u8 import (
     make_rgb_hwc_np_u8_from_rgba_hwc_np_u8
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from say_text_aloud import (
     say_text_aloud
)
from get_click_on_image import (
     get_click_on_image
)
from typing import Dict
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from pathlib import Path


def attempt_to_annotate_a_fieldgoal(
    original_path: Path,
    keypoint_id_to_instruction: Dict[str, str],
    instruct_aloud: bool = False
):
    """
    It is pretty common that we have an image,
    and we want to annotate named keypoints on it.
    This function is a helper function to do that.
    """
    while True:
        name_to_xy = {}
        for keypoint_id, instruction in keypoint_id_to_instruction.items():
            
            if instruct_aloud:
                say_text_aloud(instruction)

            rgba_hwc_np_u8 = open_as_rgba_hwc_np_u8(
                image_path=original_path,
            )

            
            rgb_hwc_np_u8 = make_rgb_hwc_np_u8_from_rgba_hwc_np_u8(
                rgba_hwc_np_u8=rgba_hwc_np_u8,
            )
            maybe_point = get_click_on_image_by_two_stage_zoom(
                # image_path=original_path,
                rgb_hwc_np_u8=rgb_hwc_np_u8,
                instructions_string=instruction
            )
            if maybe_point is not None:
                name_to_xy[keypoint_id] = maybe_point

        
        prii_named_xy_points_on_image(
            image=original_path,
            name_to_xy=name_to_xy
        )

        ans = input("type y if you happy with this, or s to skip this one, or n to redo: (N/s/y)")

        if ans.lower() in ["y", ""]:
            break

        if ans.lower() == "s":
            return None
        
        print("Please try again.")
        say_text_aloud("Please try again.")
    
    return name_to_xy
    
   
