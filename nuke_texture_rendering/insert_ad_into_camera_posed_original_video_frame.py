from typing import List
import numpy as np
import PIL
import PIL.Image
from CameraParameters import CameraParameters

from render_ads_on_subregion import (
     render_ads_on_subregion
)
from draw_euroleague_landmarks import (
     draw_euroleague_landmarks
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)

from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from get_euroleague_geometry import (
     get_euroleague_geometry
)


def insert_ad_into_camera_posed_original_video_frame(
    original_rgb_np_u8: np.ndarray,
    camera_pose: CameraParameters,
    texture_rgba_np_f32: np.ndarray,
    ad_placement_descriptors: List
) -> np.ndarray:
    """
    Sometimes you want to insert an ad (or a blend of two ad video frames)
    into an arbitrary original video frame for which you do not have the mask.
    This will overwrite the people, and that is good in some sense if you
    are trying to make an xray image from knowing the ad(s) that is/are in the LED board.
    We pluralize "ads" because there may be more than one ad at a time in the same LED board
    because the camera captures the frame during the transition between ad frames
    quite often.
    """
    assert isinstance(original_rgb_np_u8, np.ndarray)
    assert original_rgb_np_u8.ndim == 3
    assert original_rgb_np_u8.shape[2] == 3
    assert original_rgb_np_u8.dtype == np.uint8

    assert isinstance(texture_rgba_np_f32, np.ndarray)
    assert texture_rgba_np_f32.ndim == 3
    assert texture_rgba_np_f32.shape[2] == 4
    assert texture_rgba_np_f32.dtype == np.float32

    anti_aliasing_factor = 1

   
   

    shared_dir = get_the_large_capacity_shared_directory()
    geometry = get_euroleague_geometry()
    points = geometry["points"]

    do_draw_landmarks_to_prove_camera_is_good = False

    photograph_width_in_pixels = 1920
    photograph_height_in_pixels = 1080

    geometry = dict()
    geometry["points"] = points

    fake_backgrounds_dir = shared_dir / "fake_backgrounds"
    fake_backgrounds_dir.mkdir(exist_ok=True)

   


   
    if do_draw_landmarks_to_prove_camera_is_good:
        draw_euroleague_landmarks(
            original_rgb_np_u8=original_rgb_np_u8,
            camera_pose=camera_pose,
        )

    original_rgba_np_u8 = np.zeros(
        (original_rgb_np_u8.shape[0], original_rgb_np_u8.shape[1], 4),
        dtype=np.uint8
    )
    original_rgba_np_u8[:, :, :3] = original_rgb_np_u8
    original_rgba_np_u8[:, :, 3] = 255
    
    # this is a strange way to say everywhere:
    ijs = np.argwhere(original_rgba_np_u8[:, :, 3] == 255)

    for ad_placement_descriptor in ad_placement_descriptors:
        ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
    
    rgba_values_at_those_ijs = render_ads_on_subregion(
        ad_placement_descriptors=ad_placement_descriptors,
        ijs=ijs,
        photograph_width_in_pixels=photograph_width_in_pixels,  # needed to convert ijs to normalized [-1,1] x [9/16, 9/16] normalized coordinates
        photograph_height_in_pixels=photograph_height_in_pixels,
        camera_parameters=camera_pose,
    )

    # place them in 2D:
    ad_placement_accumulator = np.zeros(
        (
            1080 * anti_aliasing_factor,
            1920 * anti_aliasing_factor,
            4
        ),
        dtype=np.uint8
    )

    ad_placement_accumulator[ijs[:, 0], ijs[:, 1], :] =  rgba_values_at_those_ijs
    
    composition_rgba_np_uint8 = feathered_paste_for_images_of_the_same_size(
        bottom_layer_color_np_uint8=original_rgb_np_u8,
        top_layer_rgba_np_uint8=ad_placement_accumulator,
    )

    final_pil = PIL.Image.fromarray(composition_rgba_np_uint8)
    antialiased_pil = final_pil.resize(
        (1920, 1080),
        resample=PIL.Image.Resampling.BILINEAR
    )
    bottom_layer_color_np_uint8 = np.array(antialiased_pil)[:, :, :3]
    
    return bottom_layer_color_np_uint8

