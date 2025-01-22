from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out import (
     color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out
)
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from get_ad_name_to_paths_that_do_need_color_correction import (
     get_ad_name_to_paths_that_do_need_color_correction
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from insert_fake_ads_they_sent_to_us import (
     insert_fake_ads_they_sent_to_us
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from load_color_correction_from_json import (
     load_color_correction_from_json
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)

import numpy as np
from pathlib import Path
from prii import (
     prii
)


def check_color_calibration():



    flip_flop_dir = Path("~/ff").expanduser()
    


    out_dir = Path("~/check_color_calibration").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    assert out_dir.is_dir(), f"ERROR: {out_dir=} is not a directory."
    
    if False:
        # Uses linear light, and gets the emerald buck and ESPN red right, dark background:
        color_correction_sha256 = "4edceff5771335b7a64b1507fa1d31f38f5148f71322092c4db5ecd8ec6e985b"
        color_correction_json_path = get_file_path_of_sha256(sha256=color_correction_sha256)
        
        degree, coefficients = load_color_correction_from_json(
            json_path=color_correction_json_path
        )
    else:
        rm, r_boost = 1.36, 0.01
        gm, g_boost = 1.99, 0.01
        bm, b_boost = 2.79, 0.01
        degree = 1
        coefficients = np.array(
            [
                [ r_boost, g_boost, b_boost],
                [ rm, 0.0, 0.0],
                [0.0,  gm,  0.0],
                [0.0,  0.0,  bm],
            ]
        )
    
    # BEGIN configure which human annotations to start from, and which ones to use for what:

    clip_id_frame_index_ad_id_triplets = [
        ["slgame1", 1500, "2024-SummerLeague_Courtside_2520x126_TM_STILL"],
        ["slgame1", 2000, "SL_Tickets_CS_TM"],
        ["slgame1", 19000, "2K_CS_TM"],
        ["slgame1", 57500, "ATT_CS_TM"],
        ["slgame1", 62000, "Footlocker_CS_TM"],
        ["slgame1", 95500, "Gatorade_CS_TM"],
        ["slgame1", 103500, "Google_Pixel8_CS_TM"],
        ["slgame1", 126000, "Kia_EV9_CS_TM"],
        ["slgame1", 138500, "LVCVA_CS_TM"],
        ["slgame1", 144500, "MichelobUltra_CS_TM"],
        ["slgame1", 229500, "Microsoft_Copilot_CS_TM"],
        ["slgame1", 297000, "Panini_CS_TM"],
        ["slgame1", 297000, "NBA_Pollworker_CS_TM"],
    ]

    ad_name_to_paths_that_do_need_color_correction = get_ad_name_to_paths_that_do_need_color_correction()
    
    for clip_id, frame_index, ad_id in clip_id_frame_index_ad_id_triplets:
        camera_pose = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )

        mask_file_path = Path(f"~/r/slgame1_led/.flat/{clip_id}_{frame_index:06}_nonfloor.png").expanduser()
        original_file_path = Path(f"~/r/slgame1_led/.flat/{clip_id}_{frame_index:06}_original.jpg").expanduser()
        
        # TODO: might use original_file_path to get the original image:
        original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index,
        )

        mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
            abs_file_path=mask_file_path
        )

        ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
            clip_id=clip_id,
            with_floor_as_giant_ad=False,
            overcover_by=0.2
        )
        
   

        print("Inserting ideal ads they sent to us after color correction, noise and light and blur augmentation, etc.")
        # get the ad they sent over to us.
        # maybe augment it
        # go to linear_f32 for blurring and stuff
        # insert it

        ad_they_sent_us_file_path = ad_name_to_paths_that_do_need_color_correction[ad_id][0]
        
        ad_they_sent_us_rgb_hwc_np_u8 = \
        open_as_rgb_hwc_np_u8(ad_they_sent_us_file_path)

      
        ad_they_sent_us_rgb_hwc_np_linear_f32 = \
        convert_u8_to_linear_f32(
            ad_they_sent_us_rgb_hwc_np_u8
        )

        corrected_ad_they_sent_us_rgb_hwc_np_linear_f32 = \
        color_correct_rgb_via_polynomial_coefficients_linear_f32_in_and_out(
            degree=degree,
            coefficients=coefficients,
            rgb_hwc_np_linear_f32=ad_they_sent_us_rgb_hwc_np_linear_f32
        )

        rgb_hwc_np_u8 = insert_fake_ads_they_sent_to_us(
            original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            mask_hw_np_u8=mask_hw_np_u8,
            camera_pose=camera_pose,
            ad_placement_descriptors=ad_placement_descriptors,
            final_color_ad_rgb_np_linear_f32=corrected_ad_they_sent_us_rgb_hwc_np_linear_f32,
            verbose=False,
            flip_flop_dir=flip_flop_dir,
        )
    
        # ad insertion does not change the mask:
        rgba_hwc_np_u8 = np.concatenate(
            [
                rgb_hwc_np_u8,
                mask_hw_np_u8[:, :, np.newaxis]
            ],
            axis=2
        )
    


        fake_original_out_path = out_dir / (original_file_path.name[:-len("_original.png")] + "_fake.png")
        original_out_path = out_dir / original_file_path.name
                    
        write_rgb_hwc_np_u8_to_png(
            rgb_hwc_np_u8=rgba_hwc_np_u8[:, :, :3],
            out_abs_file_path=fake_original_out_path
        )

        # TODO: just copy the file:
        write_rgb_hwc_np_u8_to_png(
            rgb_hwc_np_u8=original_rgb_hwc_np_u8,
            out_abs_file_path=original_out_path
        )

        prii(fake_original_out_path, caption=f"{fake_original_out_path}")
        prii(original_file_path, caption=f"{original_file_path}")
        print("\n"*10)

    print("ff ~/color_calibration")


if __name__ == "__main__":
    check_color_calibration()