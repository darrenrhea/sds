from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from math import comb
from get_ad_placement_descriptor_from_jsonable import (
     get_ad_placement_descriptor_from_jsonable
)
from open_alpha_channel_image_as_a_single_channel_grayscale_image import (
     open_alpha_channel_image_as_a_single_channel_grayscale_image
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from insert_ad_into_camera_posed_original_video_frame import (
     insert_ad_into_camera_posed_original_video_frame
)
from pathlib import Path
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from prii import (
     prii
)
from CameraParameters import (
     CameraParameters
)
import better_json as bj

from color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients import (
     color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients
)


def insert_color_corrected_ad(
    # state which color correction to use:
    degree: int,
    coefficients: np.ndarray,
    # give an original image to insert into:
    original_rgb_np_u8: np.ndarray,
    camera_pose: CameraParameters,
    ad_placement_descriptor: AdPlacementDescriptor,
    uncorrected_texture_rgb_np_u8: np.ndarray,
    
    out_dir: Path,
):
    """
    Say you already fit a polynomial to color correct.
    Show how well it works on an insertion_description
    so that you can flip flop it.
    """
    assert isinstance(degree, int)
    assert degree >= 1
    assert isinstance(coefficients, np.ndarray)
    assert coefficients.shape[1] == 3
    num_coefficients_should_be = comb(degree + 3, 3)
    assert (
        coefficients.shape[0] == num_coefficients_should_be
    ), f"ERROR: {coefficients.shape=}, {num_coefficients_should_be=}"

    assert isinstance(original_rgb_np_u8, np.ndarray)
    assert original_rgb_np_u8.ndim == 3
    assert original_rgb_np_u8.shape[2] == 3
    assert original_rgb_np_u8.dtype == np.uint8
    
    assert isinstance(camera_pose, CameraParameters)

    assert isinstance(ad_placement_descriptor, AdPlacementDescriptor)

    assert isinstance(uncorrected_texture_rgb_np_u8, np.ndarray)
    assert uncorrected_texture_rgb_np_u8.ndim == 3
    assert uncorrected_texture_rgb_np_u8.shape[2] == 3
    assert uncorrected_texture_rgb_np_u8.dtype == np.uint8
    
    

    color_corrected_texture = color_correct_rgb_hwc_np_u8_image_via_polynomial_coefficients(
        degree=degree,
        coefficients=coefficients,
        rgb_hwc_np_u8=uncorrected_texture_rgb_np_u8,
    )
    prii(color_corrected_texture, caption="color_corrected_texture")
    
    
    mask_hw_np_u8 = 255 * np.zeros(
        shape=(1920, 1080),
        dtype=np.uint8,
    )

    i_min = 0
    i_max = 192
    j_min = 0
    j_max = 1536
    color_corrected_texture = color_corrected_texture[i_min:i_max, j_min:j_max, :]


    texture_rgba_np_f32 = np.zeros(
        shape=(
            color_corrected_texture.shape[0],
            color_corrected_texture.shape[1],
            4
        ),
        dtype=np.float32
    )

    texture_rgba_np_f32[:, :, :3] = color_corrected_texture
    # for figuring out the 3D alignment, this helps to make it a very different color
    # texture_rgba_np_f32[:, :, 0] = 0
    texture_rgba_np_f32[:, :, 3] = 255
    

    # We only insert one ad:
    ad_placement_descriptors = [
        ad_placement_descriptor,
    ]

    overwritten_with_its_own_ad = insert_ad_into_camera_posed_original_video_frame(
        original_rgb_np_u8=original_rgb_np_u8,
        texture_rgba_np_f32=texture_rgba_np_f32,
        ad_placement_descriptors=ad_placement_descriptors,
        camera_pose=camera_pose,
    )

    return overwritten_with_its_own_ad
   
   
