from get_a_constant_color_rgba_hwc_np_f32 import (
     get_a_constant_color_rgba_hwc_np_f32
)
from prii_nonlinear_f32 import (
     prii_nonlinear_f32
)
from prii import (
     prii
)
from insert_fake_ads_plain_version import (
     insert_fake_ads_plain_version
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from open_as_rgba_hwc_np_f32_all_channels_ranging_from_0_to_1 import (
     open_as_rgba_hwc_np_f32_all_channels_ranging_from_0_to_1
)
from pathlib import Path

from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)


def insert_fake_ads_plain_version_test():
    """
    We need an original frame segmented under led convention
    and its camera pose and the locations of the led boards
    and a texture to insert into the led boards.
    """
    
    clip_id = "slgame1"
    frame_index = 2000

    mask_file_path = Path("~/r/slgame1_led/sarah/slgame1_002000_nonfloor.png").expanduser()
    ripped_texture_image_path = Path("~/r/nba_ads_that_dont_need_color_correction/sl/005340_bg.png").expanduser()
    

    original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    prii(original_rgb_hwc_np_u8)

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )

    mask_hw_np_u8 = open_alpha_channel_image_as_a_single_channel_grayscale_image(
        abs_file_path=mask_file_path
    )

    prii(mask_hw_np_u8)


    # final_color_ad_texture_rgba_np_nonlinear_f32 = \
    # open_as_rgba_hwc_np_f32_all_channels_ranging_from_0_to_1(
    #     image_path=ripped_texture_image_path
    # )

    final_color_ad_texture_rgba_np_nonlinear_f32 = get_a_constant_color_rgba_hwc_np_f32(
        color=[1.0, 1.0, 0.0, 1.0],
        width=5120,
        height=256,
    )

    prii_nonlinear_f32(
         final_color_ad_texture_rgba_np_nonlinear_f32
    )

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.2
    )
   
    insertion_rgb = insert_fake_ads_plain_version(
        ad_placement_descriptors=ad_placement_descriptors,
        original_rgb_hwc_np_u8=original_rgb_hwc_np_u8,
        mask_hw_np_u8=mask_hw_np_u8,
        final_color_ad_texture_rgba_np_nonlinear_f32=final_color_ad_texture_rgba_np_nonlinear_f32,
        camera_pose=camera_pose,
    )
    
    prii(insertion_rgb)
   
if __name__ == "__main__":
    insert_fake_ads_plain_version_test()
