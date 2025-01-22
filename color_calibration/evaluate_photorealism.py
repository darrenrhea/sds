import pyperclip
from get_ad_id_to_frames_in_which_it_appears import (
     get_ad_id_to_frames_in_which_it_appears
)
from get_relevance_mask_from_camera_pose_and_ad_placement_descriptors import (
     get_relevance_mask_from_camera_pose_and_ad_placement_descriptors
)
import numpy as np
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_mask_path_from_clip_id_and_frame_index_and_model_id import (
     get_mask_path_from_clip_id_and_frame_index_and_model_id
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from validate_one_camera_pose import (
     validate_one_camera_pose
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from prii import (
     prii
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from show_color_correction_result import (
     show_color_correction_result
)
from pathlib import Path
from load_color_correction_from_json import (
     load_color_correction_from_json
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
def evaluate_photorealism():

    # clip_id = "dal-bos-2024-06-12-mxf"
    # model_id = "dallasfixups28"

    clip_id = "bos-dal-2024-06-09-mxf"
    model_id = "nf5l"
    
    manual = True
    if manual:
        # if clip_id == "dal-bos-2024-06-12-mxf":
        #     r_boost = 0.049
        #     g_boost = 0.038
        #     b_boost = 0.058

        #     rm = 0.48
        #     gm = 0.54
        #     bm = 0.83

        #     coefficients = np.array(
        #         [
        #             [ r_boost, g_boost, b_boost],
        #             [ rm, 0.0, 0.0],
        #             [0.0,  gm,  0.0],
        #             [0.0,  0.0,  bm],
        #         ]
        #     )
        if clip_id == "bos-dal-2024-06-09-mxf":
            # solid 000 black ads will help with determining the boosts:
            # 38,46,38

            # 216,241,254

            #20, 52, 135 
            # r_boost = 0.015
            # g_boost = 0.023
            # b_boost = 0.016

            r_boost = 0.00#7
            g_boost = 0.00#7
            b_boost = 0.00#7
            rm = 0.65
            gm = 0.63
            bm = 0.86
            # r_boost = 0.005
            # g_boost = 0.000
            # b_boost = 0.0

            # rm = 0.74
            # gm = 0.68 # 0.91
            # bm = 1.03

            
            degree = 1
            coefficients = np.array(
                [
                    [ r_boost, g_boost, b_boost],
                    [ rm, 0.0, 0.0],
                    [0.0,  gm,  0.0],
                    [0.0,  0.0,  bm],
                ]
            )
    else:
        color_correction_sha256 = "4edceff5771335b7a64b1507fa1d31f38f5148f71322092c4db5ecd8ec6e985b"
            
        color_correction_json_path = get_file_path_of_sha256(color_correction_sha256)
        #print(f"loading color correction from {color_correction_json_path}")
        
        degree, coefficients = load_color_correction_from_json(
            json_path=color_correction_json_path
        )


    print(coefficients)

    

    ad_id_to_frames_in_which_it_appears = get_ad_id_to_frames_in_which_it_appears(
        clip_id=clip_id
    )

    # we need to know the corners of the LED board in this arena:
    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0,
    )
    
    ad_placement_descriptor = ad_placement_descriptors[0]



    for ad_id, frame_indices in ad_id_to_frames_in_which_it_appears.items():
        print(f"{ad_id=}")
        for frame_index in frame_indices:
            print(f"{frame_index=}")

            original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
                clip_id=clip_id,
                frame_index=frame_index
            )

            original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
                image_path=original_file_path
            )

            mask_file_path = get_mask_path_from_clip_id_and_frame_index_and_model_id(
                clip_id=clip_id,
                frame_index=frame_index,
                model_id=model_id,
            )

            mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
                abs_file_path=mask_file_path
            )

            mask_hw_np_u8 = 255 * (mask_hw_np_u8 > 127).astype(np.uint8)

            # prii(original_rgb_np_u8)
            camera_pose = get_camera_pose_from_clip_id_and_frame_index(
                clip_id=clip_id,
                frame_index=frame_index
            )
            
            verify_camera_pose = False
            if verify_camera_pose:
                
                drawn_on = validate_one_camera_pose(
                    clip_id=clip_id,
                    frame_index=frame_index,
                )

                prii(
                    drawn_on,
                    caption="the landmarks better line up or the camera pose is wrong:"
                )


            relevance_mask = get_relevance_mask_from_camera_pose_and_ad_placement_descriptors(
                camera_pose=camera_pose,
                ad_placement_descriptors=ad_placement_descriptors,
            )
            relevance_mask = 255 - relevance_mask
            # prii(relevance_mask, caption="relevance_mask")

            mask_hw_np_u8 = np.maximum(
                mask_hw_np_u8, relevance_mask
            )
            # prii(mask_hw_np_u8, caption="mask_hw_np_u8")

            ad_file_path = Path(
                f"~/r/nba_ads/{ad_id}.jpg"
            ).expanduser()

            uncorrected_texture_rgb_np_u8 = open_as_rgb_hwc_np_u8(
                image_path=ad_file_path
            )

           
            
            out_dir = Path(
                "~/temp"
            ).expanduser()
            
            out_dir.mkdir(exist_ok=True, parents=True)

            use_linear_light = True

            # probably we should use a mask that was hand designed for regression:
            # mask_for_regression_hw_np_u8 = mask_hw_np_u8

            original_out_path = out_dir / original_file_path.name
            color_corrected_out_path = out_dir / (original_file_path.stem[:-9] + "_color_corrected.png")

            show_color_correction_result(
                use_linear_light=use_linear_light,
                degree=degree,
                coefficients=coefficients,
                original_rgb_np_u8=original_rgb_np_u8,
                camera_pose=camera_pose,
                ad_placement_descriptor=ad_placement_descriptor,
                mask_hw_np_u8=mask_hw_np_u8,
                uncorrected_texture_rgb_np_u8=uncorrected_texture_rgb_np_u8,
                original_out_path=original_out_path,
                color_corrected_out_path=color_corrected_out_path,
            )

    print("Suggest you do like:")
    suggested_command = f"ff {out_dir}"
    pyperclip.copy(suggested_command)
    print(suggested_command)
    
if __name__ == "__main__":
    evaluate_photorealism()