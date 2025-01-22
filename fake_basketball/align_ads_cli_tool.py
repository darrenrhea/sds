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
    
   

def align_ads_cli_tool():
    """
    align_ads is a command line tool that will help you match landmarks in perspective photograph
    of an ad with the same landmarks in the flat ad they sent us.
    """
    argp = argparse.ArgumentParser(
        description=textwrap.dedent(
            """\
            align_ads is a command line tool that will help you match landmarks in perspective photograph
            of an ad with the same landmarks in the flat ad they sent us.
            """
        ),
        usage=textwrap.dedent(
            """\
            align_ads
            """
        )
    )
    
    opt = argp.parse_args()
    clip_id = "slgame1"
    ad_id = "2024-SummerLeague_Courtside_2520x126_TM_STILL"
    ad_path = Path("~/r/nba_ads/sl/Thomas_Mack/2024-SummerLeague_Courtside_2520x126_TM_STILL.png").expanduser()
    frame_index = 1544

    original_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    
    src_rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(original_path)
    prii(
        ad_path,
        caption=f"This is the ad called {ad_id}"
    )
    prii(
        src_rgb_hwc_np_u8,
        caption=f"This better show the ad called {ad_id}"
    )

    # name_to_xy = attempt_to_annotate_a_quad(
    #     image_path=original_path,
    #     dst_height=out_height,
    #     dst_width=out_width,
    # )

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

    # prii_named_xy_points_on_image_with_auto_zoom(
    #     image=original_path,
    #     name_to_xy=name_to_xy
    # )

    # rip = rip_quad_out_of_image(
    #     src_image=src_rgb_hwc_np_u8,
    #     name_to_xy=name_to_xy,
    #     dst_height=out_height,
    #     dst_width=out_width,
    # )


    # out_dir = Path("~/r/nba_ads_that_dont_need_color_correction").expanduser() / ad_id
    
    # out_dir.mkdir(parents=True, exist_ok=True)

    # out_abs_file_path = out_dir / f"{ad_id}_from_{clip_id}_{frame_index}.png"

    # write_rgb_hwc_np_u8_to_png(
    #     rgb_hwc_np_u8=rip,
    #     out_abs_file_path=out_abs_file_path
    # )

    # put_on_clipboard = f"gimp {out_abs_file_path}"
    # pyperclip.copy(put_on_clipboard)
    # print(f"We suggest you run:\n\n   {put_on_clipboard}")
    # print("It is already on your clipboard, just paste it into your terminal and hit enter.")

if __name__ == "__main__":
    align_ads_cli_tool()
