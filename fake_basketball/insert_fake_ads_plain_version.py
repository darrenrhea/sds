from typing import List
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from CameraParameters import (
     CameraParameters
)
from insert_quads_into_camera_posed_image_behind_mask import (
     insert_quads_into_camera_posed_image_behind_mask
)
import numpy as np


def insert_fake_ads_plain_version(

    ad_placement_descriptors: List[AdPlacementDescriptor],
    original_rgb_hwc_np_u8: np.ndarray,
    mask_hw_np_u8: np.ndarray,  # a mask to limit which pixels can be changed by insertion of ads
    final_color_ad_texture_rgba_np_nonlinear_f32: np.ndarray,
    camera_pose: CameraParameters,
):
    """
    This inserts ads in a plain manner, i.e. the color you send in is what is used.
    We will use it for rips, ads that don't need color correction.
    """
  
 
    # if there are several ad boards, we stick the same texture in all for now:
    textured_ad_placement_descriptors = []
    for ad_placement_descriptor in ad_placement_descriptors:
        ad_placement_descriptor.texture_rgba_np_f32 = final_color_ad_texture_rgba_np_nonlinear_f32
        textured_ad_placement_descriptors.append(ad_placement_descriptor)

    composition_rgb_np_u8 = insert_quads_into_camera_posed_image_behind_mask(
        use_linear_light=False,
        original_rgb_np_u8=original_rgb_hwc_np_u8,
        mask_hw_np_u8=mask_hw_np_u8,
        camera_pose=camera_pose,
        textured_ad_placement_descriptors=textured_ad_placement_descriptors,
        anti_aliasing_factor=1,
    )

    assert composition_rgb_np_u8.dtype == np.uint8
    assert composition_rgb_np_u8.shape[0] == original_rgb_hwc_np_u8.shape[0]
    assert composition_rgb_np_u8.shape[1] == original_rgb_hwc_np_u8.shape[1]
    assert composition_rgb_np_u8.shape[2] == 3
    return composition_rgb_np_u8