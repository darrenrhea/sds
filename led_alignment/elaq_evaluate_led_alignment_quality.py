import sys
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from report_homography_error import (
     report_homography_error
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
import pyperclip
from find_homography_from_2d_correspondences import (
     find_homography_from_2d_correspondences
)
from pathlib import Path
from prii import (
     prii
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import textwrap
import numpy as np
import cv2
import better_json as bj


def get_ad_rgb_hwc_np_u8_by_ad_id(ad_id):
    ad_path = Path(
        f"~/r/nba_ads/summer_league_2024/{ad_id}.png"
    ).expanduser()

    ad = open_as_rgb_hwc_np_u8(ad_path)
    return ad


def get_all_sets_of_frame_ad_2d_correspondences():
    """
    This function is a generator that yields all the frame-ad correspondences.
    """
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
    all_sets = []
    for p in save_dir.glob("*.json"):
        jsonable = bj.load(p)
        try:
            clip_id, frame_index, ad_id = jsonable["clip_id"], jsonable["frame_index"], jsonable["ad_id"]
        except KeyError:
            print(f"Something is wrong with {p}")
            sys.exit(1)
        
        jsonable["abs_file_path_str"] = str(p.resolve())

        all_sets.append(jsonable)
    return all_sets


def get_all_sets_of_frame_ad_2d_correspondences_for_this_clip_id(clip_id):
    return [
        x for x in get_all_sets_of_frame_ad_2d_correspondences() if x["clip_id"] == clip_id
    ]


def get_all_sets_of_frame_ad_2d_correspondences_for_this_ad_id(ad_id):
    return [
        x for x in get_all_sets_of_frame_ad_2d_correspondences() if x["ad_id"] == ad_id
    ]

 
def elaq_evaluate_led_alignment_quality(
    ad_id: str,
) -> None:
    """
    Shows the quality of the alignment of the ad they gave us with the video frame.
    """
    all_matches = get_all_sets_of_frame_ad_2d_correspondences_for_this_ad_id(ad_id)
    # if len(all_matches) >= 2:
    #     print(f"Found more than one 2D-2D correspondence file for {ad_id}")
    #     for match in all_matches:
    #         print(match["abs_file_path_str"])
    #         sys.exit(1)
    
    for match in all_matches:
        clip_id = match["clip_id"]
        frame_index = match["frame_index"]
        ad_id = match["ad_id"]
        abs_file_path_str = match["abs_file_path_str"]
        
        ad = get_ad_rgb_hwc_np_u8_by_ad_id(
            ad_id=ad_id
        )
       
        frame = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )

        print("Please look at the two images that we have registered 2D-2D point correspondences between, the ad and the video frame:")
       

        correspondences = match["correspondences"]
        dst_points = np.array([v["frame"] for k, v in correspondences.items()])
        src_points = np.array([v["ad"] for k, v in correspondences.items()])

        named_frame_points = {k: v["frame"] for k, v in correspondences.items()}
        named_ad_points = {k: v["ad"] for k, v in correspondences.items()}
        names = list(named_frame_points.keys())
        
        print("\nThe ad they sent us with named landmarks/keypoints:")
        prii_named_xy_points_on_image(
            name_to_xy=named_ad_points,
            image=ad,
            output_image_file_path= None,
            default_color=(0, 255, 0),  # green is the default
            dont_show=False,
        )

        print("\nThe video frame with named landmarks/keypoints:")
        prii_named_xy_points_on_image(
            name_to_xy=named_frame_points,
            image=frame,
            output_image_file_path= None,
            default_color=(0, 255, 0),  # green is the default
            dont_show=False,
        )

       

        answer = find_homography_from_2d_correspondences(
            src_points=src_points,
            dst_points=dst_points,
            ransacReprojThreshold=20
        )

        assert answer["success"]
        homography_3x3 = answer["homography_3x3"]

        # We are hopeful that, since it was annotated by human, the points all get used.
        # matched_dst_points = answer["matched_dst_points"]
        # matched_src_points = answer["matched_src_points"]
        # match_indicator = answer["match_indicator"]

        out_height, out_width = frame.shape[:2]

        new_image = np.zeros(shape=(out_height, out_width, 4), dtype=np.uint8)
        ad_rgba = np.zeros(shape=(ad.shape[0], ad.shape[1], 4), dtype=np.uint8)
        ad_rgba[:, :, :3] = ad
        ad_rgba[:, :, 3] = 255
        cv2.warpPerspective(src=ad_rgba, dst=new_image, M=homography_3x3, dsize=(out_width, out_height))
        
        prii(new_image)

        overwritten_with_its_own_ad = feathered_paste_for_images_of_the_same_size(
              top_layer_rgba_np_uint8=new_image,
              bottom_layer_color_np_uint8=frame,
        )


        out_dir = Path(
            "~/tffd_temp_flip_flop_dir"
        ).expanduser()
        out_dir.mkdir(exist_ok=True, parents=True)
        original_out_path = out_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
        color_corrected_out_path = out_dir / f"{clip_id}_{frame_index:06d}_insertion.jpg"
        prii(
            frame,
            caption=f"this is the original video frame, {clip_id}_{frame_index:06d}_original.jpg:",
            out=original_out_path
        )
    
        prii(
            overwritten_with_its_own_ad,
            caption="this is augmented with its own ad without any color correction:",
            out=color_corrected_out_path
        )

        report_homography_error(
            matched_source_points=src_points,
            homography_3x3=homography_3x3,
            matched_destination_points=dst_points,
            names=names
        )
        
        print(f"You can see the 2D-2D data in {abs_file_path_str}")

        s = f"flipflop {out_dir}"
        print(s)
        pyperclip.copy(s)




if __name__ == "__main__":
    elaq_evaluate_led_alignment_quality()