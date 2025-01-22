from prii import (
     prii
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from get_clip_id_and_frame_index_from_file_name import (
     get_clip_id_and_frame_index_from_file_name
)

from colorama import Fore, Style
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from get_clicks_on_image import ( 
     get_clicks_on_image
)

from pathlib import Path
import better_json as bj
import numpy as np


def attempt_to_annotate_an_led_screen_occluding_object(
    cutout_path: Path,
):
    file_name = cutout_path.name

    clip_id, frame_index = get_clip_id_and_frame_index_from_file_name(
        file_name
    )

    original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    while True:
        rgba_hwc_np_u8 = open_as_rgba_hwc_np_u8(original_file_path)
        alpha_source = open_as_rgba_hwc_np_u8(cutout_path)
        # 0 -> 64, 255 -> 255
        rgba_hwc_np_u8[:, :, 3] = 128 + alpha_source[:, :, 3] // 2
        rgba_hwc_np_u8[:, :, 2] = np.clip(0, 255,
            (
                rgba_hwc_np_u8[:, :, 2].astype(np.uint16)
                +
                alpha_source[:, :, 3].astype(np.uint16)
            ) // 2
        ).astype(np.uint8)

        prii(rgba_hwc_np_u8, caption="Here it is:")
        points = get_clicks_on_image(
            image_path=None,
            rgba_hwc_np_u8=rgba_hwc_np_u8,
            instructions_string="Click on the bottom of the LED region, then the top of the LED region"
        )

        if len(points) != 2:
            print(f"You were supposed to click 2 points, but you clicked {len(points)} times!. Try again.")
            continue
        
        # If we get here, we have 2 points.
        name_to_xy = {
            "bottom_of_led_screen": points[0],
            "top_of_led_screen": points[1],
        }
        
        prii_named_xy_points_on_image(
            image=cutout_path,
            name_to_xy=name_to_xy
        )

        ans = input("type y if you happy with this, or s to skip this one, or n to redo: (N/s/y)")

        if ans.lower() == "y":
            break

        if ans.lower() == "s":
            return None
        
        print("Try again please:")
    
    return name_to_xy
    
   

 
if __name__ == "__main__":
    kind = "led_screen_occluding_object"
    subfolder = "objects"

    dir_of_one_kind = Path(
        f"~/r/nba_misc_cutouts_approved/{subfolder}"
    ).expanduser()

    cutout_paths = list(
        dir_of_one_kind.glob("*.png")
    )

    print(f"Going to annotate {len(cutout_paths)} cutouts of {kind=}, namely:")
    
    for cutout_path in cutout_paths:
        print(cutout_path)

    for cutout_path in cutout_paths:
        print(f"{cutout_path=}")
        out_path = cutout_path.with_suffix(".json")
        print(f"{out_path=}")
        if out_path.exists():
            print(f"skipping {cutout_path} because {out_path} already exists")
            continue

        metainfo = dict(
            kind=kind,
            name_to_xy={}
        )


        name_to_xy = attempt_to_annotate_an_led_screen_occluding_object(cutout_path)

        if name_to_xy is not None:
            metainfo["name_to_xy"] = name_to_xy
            bj.color_print_json(metainfo)
            bj.dump(obj=metainfo, fp=out_path)
            print(f"wrote {out_path}")
