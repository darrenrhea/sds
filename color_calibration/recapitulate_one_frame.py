from prii import (
     prii
)
from insert_color_corrected_ad import (
     insert_color_corrected_ad
)
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from get_ad_placement_descriptor_from_jsonable import (
     get_ad_placement_descriptor_from_jsonable
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import pyperclip
from load_color_correction_from_json import (
     load_color_correction_from_json
)
from pathlib import Path

import better_json as bj

import argparse

def recapitulate_one_frame():
    """
    Given a clip_id and a frame_index and human knowledge of which ad is being displayed at that time,
    this re-synthesizes the frame by inserting the ad into the frame.

    Then it compares that recreation to reality, to look for color correction errors,
    and bad noise models, like failure to look as "ringing" and "artifact-y" as the original.
    Some invokations to try:
    python recapitulate_one_frame.py --clip_id bos-mia-2024-04-21-mxf --frame_index 495500 --ad_id NBA_ID && flipflop ~/temp

    """
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "--clip_id",
        required=True,
        help="The clip_id of the video",
    )
    argp.add_argument(
        "--frame_index",
        required=True,
        type=int,
        help="The frame index of the video",
    )
    argp.add_argument(
        "--ad_id",
        required=True,
        help="The insertion_description_id of the ad",
    )
    opt = argp.parse_args()
    clip_id = opt.clip_id
    frame_index = opt.frame_index
    ad_id = opt.ad_id

    # led_image_path = get_file_path_of_sha256(
    #     sha256=led_image_sha256
    # )
    led_image_path = Path(
        f"~/r/nba_ads/{ad_id}.jpg"
    ).expanduser()

    uncorrected_texture_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=led_image_path
    )


    # BEGIN way of getting the LED corners:
    insertion_description_id_to_use_for_led_corners = {
        # "0e4d1982-baeb-4b04-8c18-774f3bce4084",  # Playoffs_Title  NBA Playoffs Presented by Google Pixel, bos-mia-2024-04-21-mxf_365000
        # "41bb4d0b-d2cf-4d15-8482-abb6493520ba",  # Playoffs_Title  NBA Playoffs Presented by Google Pixel, bos-mia-2024-04-21-mxf_365500
        # "e8d4d8d2-0409-4eb1-a993-f0a7d8ce58ab",  # NBA_APP_MSFT       get the App powered by microsoft bos-mia-2024-04-21-mxf_397000
        "ESPN_DAL_LAC_NEXT_ABC": "60820db1-9028-4566-86a0-84d6056168fb",  #    Black version Playoffs Dallas Mavericks x LA Clippers COMING UP NEXT abc bos-mia-2024-04-21-mxf_412000
        "ESPN_NBA_Finals": "cbc658ec-a9c3-4f67-8d07-667c03a35f27",  # ESPN_NBA_Finals     abc home of the NBA Finals begin june 6 bos-mia-2024-04-21-mxf_440000
        # "fd217680-07d7-4f0f-a8ff-8a2bdf2a25e4",  # different_here    Green Different Here bos-mia-2024-04-21-mxf_471000
        "NBA_ID": "e4e347df-b49a-4f00-8990-d5d7489b0812",  #       NBA ID Sign up. Score. bos-mia-2024-04-21-mxf_501500
        # "c275ba31-0faf-4564-9134-a1e1d03e8805",  # NHL_Playoffs      NHL Stanley Cup playoffs bos-mia-2024-04-21-mxf_539000
        # "48964eb8-573e-4455-9db6-e6874c66ef62",  # ESPN_APP GET ESPN+ bos-mia-2024-04-21-mxf_557000
        # "2a04b7dd-8d83-4455-927e-002b16b11128",  # ESPN_MIL_IND_FRI     Milwaukee Bucks x Indiana Pacers bos-mia-2024-04-21-mxf_628500
        # "20f0eff2-34e0-4921-96d2-7f2b83ff1b7a",  # NBA_Store     NBA Store bos-mia-2024-04-21-mxf_657000
        # "1b9f9cd9-0d15-4965-ab74-a6b2626dbd23",  # Playoffs_PHI_NYK_TOM_TNT      NBA Playoffs 76ers versus Knicks bos-mia-2024-04-21-mxf_712000
        # "c93bd561-f628-4043-86d0-9c601ce23993",  # PickEm_Bracket_Challenge Pickem Bracket Challenge Play now bos-mia-2024-04-21-mxf_770500
        # "7466b2fe-0b71-437e-a5d4-5189f968469c",  # Playoffs_DAL_LAC_NEXT_ABC white version of Playoffs Dallas Mavericks x LA Clippers NEXT
    }[ad_id]

    insertion_desc = bj.load(
        f"~/r/color_correction_data/insertion_descriptions/{insertion_description_id_to_use_for_led_corners}.json5"
    )

    ad_placement_descriptor_jsonable = insertion_desc["ad_placement_descriptor"]

    ad_placement_descriptor = get_ad_placement_descriptor_from_jsonable(
        ad_placement_descriptor_jsonable=ad_placement_descriptor_jsonable
    )
    assert isinstance(ad_placement_descriptor, AdPlacementDescriptor)
    # ENDOF way of getting the LED corners.
    
    color_correction_sha256 = "bd545cba8ac10558b8a5a4eeba40bc3be9f1e809975fd7e6ad38d6a3ac598140"
    color_correction_json_path = get_file_path_of_sha256(color_correction_sha256)
    print(f"loading color correction from {color_correction_json_path}")
    
    degree, coefficients = load_color_correction_from_json(
        json_path=color_correction_json_path
    )
    
    out_dir = Path(
        "~/temp"
    ).expanduser()
    
    out_dir.mkdir(exist_ok=True, parents=True)


    original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=original_file_path
    )

    
    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    ) 

    prii(uncorrected_texture_rgb_np_u8, caption="original_texture_they_sent_over")

    overwritten_with_its_own_ad = insert_color_corrected_ad(
        degree=degree,
        coefficients=coefficients,
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        ad_placement_descriptor=ad_placement_descriptor,
        uncorrected_texture_rgb_np_u8=uncorrected_texture_rgb_np_u8,
        out_dir=out_dir,
    )

    original_out_path = out_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
    color_corrected_out_path = out_dir / f"{clip_id}_{frame_index:06d}_corrected.jpg"

    prii(
        original_rgb_np_u8,
        caption=f"this is the original video frame, {clip_id}_{frame_index:06d}_original.jpg:",
        out=original_out_path
    )
   
    prii(
        overwritten_with_its_own_ad,
        caption="this is augmented with its own ad without color correction:",
        out=color_corrected_out_path
    )
   
    s = "flipflop ~/temp"
    pyperclip.copy(s)
    print("We suggest you run the following command:")
    print(s)
    print("you can just paste since it is on the clipboard")
 

if __name__ == "__main__":
    recapitulate_one_frame()