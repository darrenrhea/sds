from typing import List
import PIL.Image
import numpy as np
from CameraParameters import CameraParameters
from render_ads_on_subregion import (
     render_ads_on_subregion
)
from get_euroleague_geometry import (
     get_euroleague_geometry
)




def make_relevance_mask_for_led_boards(
    camera_posed_original_video_frame: dict,
    ad_placement_descriptors: List
) -> np.ndarray:
    """
    See also get_relevance_mask_from_camera_pose_and_ad_placement_descriptors
    
    Sometimes you want to insert an ad (or a blend of two ad video frames)
    into an arbitrary original video frame for which you do not have the mask.
    This will overwrite the people, and that is good in some sense if you
    are trying to make an xray image from knowing the ad(s) that is/are in the LED board.
    We pluralize "ads" because there may be more than one ad at a time in the same LED board
    because the camera captures the frame during the transition between ad frames
    quite often.
    """
    assert isinstance(camera_posed_original_video_frame, dict)
    assert "clip_id" in camera_posed_original_video_frame
    assert "frame_index" in camera_posed_original_video_frame
    assert "original_file_path" in camera_posed_original_video_frame
    assert "camera_pose" in camera_posed_original_video_frame

    camera_parameters = camera_posed_original_video_frame["camera_pose"]
    assert isinstance(camera_parameters, CameraParameters)


    anti_aliasing_factor = 1

    geometry = get_euroleague_geometry()
    points = geometry["points"]


    photograph_width_in_pixels = 1920
    photograph_height_in_pixels = 1080

    geometry = dict()
    geometry["points"] = points

    # this is a strange way to say everywhere:
    temp = np.zeros(
        shape=(
            1080 * anti_aliasing_factor,
            1920 * anti_aliasing_factor
        ),
        dtype=np.uint8
    )
    ijs = np.argwhere(temp == 0)

    texture_rgba_np_f32 = np.zeros(
        (100, 100, 4),
        dtype=np.float32
    )
    
    texture_rgba_np_f32[...] = 255.0

    for ad_placement_descriptor in ad_placement_descriptors:
        ad_placement_descriptor.texture_rgba_np_f32 = texture_rgba_np_f32
    
    rgba_values_at_those_ijs = render_ads_on_subregion(
        ad_placement_descriptors=ad_placement_descriptors,
        ijs=ijs,
        # needed to convert ijs to normalized [-1,1] x [9/16, 9/16] normalized coordinates:
        photograph_width_in_pixels=photograph_width_in_pixels * anti_aliasing_factor,  
        photograph_height_in_pixels=photograph_height_in_pixels * anti_aliasing_factor,
        camera_parameters=camera_parameters,
    )
    # place them in 2D:
    ad_placement_accumulator = np.zeros(
        shape=(
            1080 * anti_aliasing_factor,
            1920 * anti_aliasing_factor,
            4
        ),
        dtype=np.uint8
    )

    ad_placement_accumulator[ijs[:, 0], ijs[:, 1], :] =  rgba_values_at_those_ijs

    relevance_mask = ad_placement_accumulator[:, :, 3].copy()

    if anti_aliasing_factor == 1:
        return relevance_mask.copy()
    else:
        relevance_pil = PIL.Image.fromarray(relevance_mask)
        resized_pil = relevance_pil.resize(
            (
                1920,
                1080,
            ),
            resample=PIL.Image.Resampling.BILINEAR
        )
        relevance_final = np.array(resized_pil)
        assert relevance_final.dtype == np.uint8
        return relevance_final

