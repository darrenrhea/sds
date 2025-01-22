from rip_quad_out_of_image import (
     rip_quad_out_of_image
)
from prii_named_xy_points_on_image_with_auto_zoom import (
     prii_named_xy_points_on_image_with_auto_zoom
)
import pyperclip
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from typing import Optional
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_clicks_on_image import (
     get_clicks_on_image
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from prii import (
     prii
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)

import argparse

import textwrap


from pathlib import Path

import numpy as np



def attempt_to_annotate_a_quad(
    image_path: Path,
    dst_height: int,
    dst_width: int,
) -> Optional[np.ndarray]:
    while True:
        points = get_clicks_on_image(
            image_path=image_path,
            rgba_hwc_np_u8=None,
            instructions_string="click the tl then tr then br the bl"
        )

        rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(image_path)

        if len(points) != 4:
            print(f"You were supposed to click 4 points, but you clicked {len(points)} times!. Please try again.")
            continue
        
        # If we get here, we have 4 points.
        name_to_xy = {
            "tl": points[0],
            "tr": points[1],
            "br": points[2],
            "bl": points[3],
        }
        
        prii_named_xy_points_on_image(
            image=image_path,
            name_to_xy=name_to_xy
        )

        rip = rip_quad_out_of_image(
            src_image=rgb_hwc_np_u8,
            name_to_xy=name_to_xy,
            dst_height=dst_height,
            dst_width=dst_width,
        )

        prii(rip)

        

        ans = input("type y if you happy with this, or s to skip this one, or n to redo: (N/s/y)")

        if ans.lower() == "y":
            break

        if ans.lower() == "s":
            return None
        
        print("Try again please:")
    
    return name_to_xy
    
   

def aifar_annotate_images_for_ad_rips_cli_tool():
    """
    aifar is used to annotate the 4 corners of an image to use as a ad rip.
    ad_rip, advertisement rip, is a term used in the NBA to refer to a still image.

    """
    argp = argparse.ArgumentParser(
        description="annotate 4 corners of an image to use as a ad rip",
        usage=textwrap.dedent(
            """\
            aifar_annotate_images_for_ad_rips <clip_id> <frame_index>

            Do something like this:
            
            
            aifar_annotate_images_for_ad_rips \\
            --out_width 1856 \\
            --out_height 256 \\
            --clip_id slgame1 \\
            --ad_id 2024-SummerLeague_Courtside_2520x126_TM_STILL \\
            --frame_index 1800
            """
        )
    )
   
    argp.add_argument(
        "--clip_id",
        type=str,
        required=True,
    )
    
    argp.add_argument(
        "--out_width",
        type=int,
        default=1024,
    )
    argp.add_argument(
        "--out_height",
        type=int,
        default=144,
    )

    argp.add_argument("--ad_id", type=str, required=True)
    argp.add_argument("--frame_index", required=True, type=int)
    
    opt = argp.parse_args()
    clip_id = opt.clip_id
    ad_id = opt.ad_id
    out_width = opt.out_width
    out_height = opt.out_height
    frame_index = opt.frame_index

    original_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    
    src_rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(original_path)

    prii(
        src_rgb_hwc_np_u8,
        caption=f"This better show the ad called {ad_id}"
    )

    name_to_xy = attempt_to_annotate_a_quad(
        image_path=original_path,
        dst_height=out_height,
        dst_width=out_width,
    )

    # name_to_xy = {
    #     "tl": [
    #         1335.0,
    #         299.0
    #     ],
    #     "tr": [
    #         1832.0,
    #         315.0
    #     ],
    #     "br": [
    #         1827.0,
    #         372.0
    #     ],
    #     "bl": [
    #         1334.0,
    #         352.0
    #     ]
    # }

    prii_named_xy_points_on_image_with_auto_zoom(
        image=original_path,
        name_to_xy=name_to_xy
    )

    rip = rip_quad_out_of_image(
        src_image=src_rgb_hwc_np_u8,
        name_to_xy=name_to_xy,
        dst_height=out_height,
        dst_width=out_width,
    )


    out_dir = Path("~/r/nba_ads_that_dont_need_color_correction").expanduser() / ad_id
    
    out_dir.mkdir(parents=True, exist_ok=True)

    out_abs_file_path = out_dir / f"{ad_id}_from_{clip_id}_{frame_index:06d}.png"

    write_rgb_hwc_np_u8_to_png(
        rgb_hwc_np_u8=rip,
        out_abs_file_path=out_abs_file_path
    )

    put_on_clipboard = f"gimp {out_abs_file_path}"
    pyperclip.copy(put_on_clipboard)
    print(f"We suggest you run:\n\n   {put_on_clipboard}")
    print("It is already on your clipboard, just paste it into your terminal and hit enter.")

    
