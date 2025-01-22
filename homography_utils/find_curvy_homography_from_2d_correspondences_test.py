from map_points_through_curvy_homography import (
     map_points_through_curvy_homography
)
from report_curvy_homography_error import (
     report_curvy_homography_error
)
from report_homography_error import (
     report_homography_error
)
from find_curvy_homography_from_2d_correspondences import (
     find_curvy_homography_from_2d_correspondences
)
from get_all_sets_of_frame_ad_2d_correspondences_for_this_clip_id import (
     get_all_sets_of_frame_ad_2d_correspondences_for_this_clip_id
)
from get_ad_rgb_hwc_np_u8_by_ad_id import (
     get_ad_rgb_hwc_np_u8_by_ad_id
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from prii_xy_points_on_image import (
     prii_xy_points_on_image
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
import numpy as np
import cv2




def test_find_homography_from_2d_correspondences_1():

    clip_id = "slgame1"
    # frame_index = 1544
    frame_index = 1892
    all_matches = get_all_sets_of_frame_ad_2d_correspondences_for_this_clip_id(
        clip_id=clip_id
    )
    all_matches = [
        x for x in all_matches if x["frame_index"] == frame_index
    ]

    for match in all_matches:
        ad_id = match["ad_id"]
        
        ad = get_ad_rgb_hwc_np_u8_by_ad_id(
            ad_id=ad_id
        )
       
        frame = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )

        print("Please look at the two images that we want to register points between")
        prii(ad)
        prii(frame)

        correspondences = match["correspondences"]
        dst_points = np.array([v["frame"] for k, v in correspondences.items()])
        src_points = np.array([v["ad"] for k, v in correspondences.items()])

        prii_xy_points_on_image(
            xys=src_points,
            image=ad,
            output_image_file_path= None,
            default_color=(0, 255, 0),  # green is the default
            dont_show=False,
        )

        prii_xy_points_on_image(
            xys=dst_points,
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
        matched_dst_points = answer["matched_dst_points"]
        matched_src_points = answer["matched_src_points"]
        match_indicator = answer["match_indicator"]



        print(f"homography_3x3=\n{homography_3x3}")
        print(f"matched_src_points=\n{matched_src_points}")
        print(f"matched_dst_points=\n{matched_dst_points}")
        print(f"match_indicator=\n{match_indicator}")
        
        names = [f"{i}" for i in range(len(src_points))]
        
        report_homography_error(
            matched_source_points=src_points,
            homography_3x3=homography_3x3,
            matched_destination_points=dst_points,
            names=names
        )

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
            "~/uncorrected"
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

        homography_3x3 = answer["homography_3x3"]
        matched_dst_points = answer["matched_dst_points"]
        matched_src_points = answer["matched_src_points"]
        match_indicator = answer["match_indicator"]
        
        numerator_params, denominator_params = find_curvy_homography_from_2d_correspondences(
            src_points=src_points,
            dst_points=dst_points,
            initial_guess_for_homography_3x3=homography_3x3,
            learning_rate=0.00000000000001,
            max_iterations=400000,
        )

        print("Found the curvy homography to be:")
        print(f"{numerator_params=}")
        print(f"{denominator_params=}")


        names = [f"point_{i}" for i in range(len(src_points))]
            
        report_curvy_homography_error(
            matched_source_points=src_points,
            matched_destination_points=dst_points,
            numerator_params=numerator_params,
            denominator_params=denominator_params,
            names=names
        )
        points = np.zeros(
            (10000, 2),
            dtype=np.float32
        )
        points[:, 0] = np.linspace(0, ad.shape[1], 10000)

        predictions = map_points_through_curvy_homography(
            points=points, 
            numerator_params=numerator_params,
            denominator_params=denominator_params,
        )

        prii_xy_points_on_image(
            xys=predictions,
            image=frame,
            output_image_file_path= None,
            default_color=(0, 255, 0),  # green is the default
            dont_show=False,
        )


        s = f"flipflop {original_out_path} {color_corrected_out_path}"
        print(s)
        pyperclip.copy(s)


if __name__ == "__main__":
    test_find_homography_from_2d_correspondences_1()
    print("find_homography_from_2d_correspondences_test.py has run successfully")