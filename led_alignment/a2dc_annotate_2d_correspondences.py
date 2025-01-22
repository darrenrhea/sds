from load_json_file import (
     load_json_file
)
from color_print_json import (
     color_print_json
)
import textwrap
from save_jsonable import (
     save_jsonable
)
from get_click_on_image_by_two_stage_zoom import (
     get_click_on_image_by_two_stage_zoom
)
from pathlib import Path
from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)


def a2dc_annotate_2d_correspondences(
    ad_id: str,
    clip_id: str,
    frame_index: int,
):

    # look at these two images that we want to register points between

    # almost any monitor can fit a window of 800x800 pixels, so:
    max_display_width = 800
    max_display_height = 800

    save_dir = Path(
        "~/r/keypoint_correspondences_data"
    ).expanduser()

    assert (
        save_dir.exists()
    ), textwrap.dedent(
        f"""\
        {save_dir=} does not exist, maybe you need to clone it?"
        (cd ~/r && git clone git@github.com:darrenrhea/keypoint_correspondences)
        """
    )


    json_path = save_dir / f"matches_between{clip_id}_{frame_index:06d}_and_{ad_id}.json"

    if json_path.exists():
        print(f"{json_path} already exists, and defines these points:")
        jsonable = load_json_file(json_path)
        color_print_json(jsonable)
        correspondences = jsonable["correspondences"]
    else:
        correspondences = {}

    ad_path = Path(
        f"~/r/nba_ads/sl/{ad_id}.png"
    ).expanduser()

    ad = open_as_rgb_hwc_np_u8(ad_path)

    frame = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )






    while True:

        print("Please look at these two images that we want to register points between, the LED ad and the video frame:")
        print("then come-up-with / name / dub some landmark/keypoint that you can see in both images.")
        prii(ad)
        prii(frame)

        if len(correspondences) > 0:
            print("So far you have defined these landmarks:")
            for key in correspondences:
                print(key)    
        
        landmark_name = input("Either enter a name for a landmark, or press enter to stop annotating matches\n\n>>> ")
        if landmark_name == "":
            break

        print(f'Now please click on the point that you have dubbed "{landmark_name}" in the two images')
        print("via two-stage-zoom-in-click-refinement, first the video frame:")



        frame_click = get_click_on_image_by_two_stage_zoom(
            max_display_width=max_display_width,
            max_display_height=max_display_height,
            rgb_hwc_np_u8=frame,
            instructions_string=f"click on the point you called {landmark_name} in the frame",
        )

        ad_click = get_click_on_image_by_two_stage_zoom(
            max_display_width=max_display_width,
            max_display_height=max_display_height,
            rgb_hwc_np_u8=ad,
            instructions_string=f"click on the point you called {landmark_name} in the ad",
        )

        correspondences[landmark_name] = {
            "frame": frame_click,
            "ad": ad_click
        }

    obj = {
        "clip_id": clip_id,
        "frame_index": frame_index,
        "ad_id": ad_id,
        "correspondences": correspondences,
    }

    save_jsonable(
        obj=obj,
        fp=json_path,
    )


    print(f"bat {json_path}")


