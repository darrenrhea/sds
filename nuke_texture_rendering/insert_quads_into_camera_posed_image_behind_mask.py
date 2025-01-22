from feathered_paste_for_images_of_the_same_size_linear_f32 import (
     feathered_paste_for_images_of_the_same_size_linear_f32
)
from prii import (
     prii
)
from prii_linear_f32 import (
     prii_linear_f32
)
from typing import List
import numpy as np

from CameraParameters import CameraParameters
from render_ads_on_subregion import (
     render_ads_on_subregion
)
from feathered_paste_for_images_of_the_same_size import (
     feathered_paste_for_images_of_the_same_size
)
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from typing import Optional


def insert_quads_into_camera_posed_image_behind_mask(
    use_linear_light: bool,
    camera_pose: CameraParameters,
    mask_hw_np_u8: np.ndarray,
    textured_ad_placement_descriptors: List[AdPlacementDescriptor],
    anti_aliasing_factor: int = 1,
    original_rgb_np_linear_f32: Optional[np.ndarray] = None,
    original_rgb_np_u8: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Given an image with a well-fitted camera-pose,
    this will draw 3D-perspectivized flat quads onto the image
    BUT masked by mask_hw_np_u8, i.e. the 255 white part of the mask completely forbids drawing.
    Sole consumer of this procedure is
    gabofa_generate_a_bunch_of_fake_annotations around line 443.

    Each texture_rgba_np_f32 should already be color-corrected
    and augmented, say by noise,
    before calling this, because the texture's rgb color are inserted as is.

    Apologies:
    Don't use this for placing lettered ads on the floor.

    The order of stack of "transparency films" or layers is wrong for that purpose.

    This is more designed for:
    * ad insertions being rendered into the LED board "holes"
    * inserting a different floor texture underneath the people
    * inserting the blurry mirror image of the floor into the ad board hole.
    """
    if use_linear_light:
        assert original_rgb_np_linear_f32 is not None, "if you are use_linear_light, original_rgb_np_linear_f32 should be given"
        assert isinstance(original_rgb_np_linear_f32, np.ndarray)
        assert original_rgb_np_linear_f32.ndim == 3
        assert original_rgb_np_linear_f32.dtype == np.float32
        assert original_rgb_np_linear_f32.shape[2] == 3
        assert np.min(original_rgb_np_linear_f32) >= -0.01
        assert np.max(original_rgb_np_linear_f32) <= 1.01
        photograph_height_in_pixels, photograph_width_in_pixels, _ = original_rgb_np_linear_f32.shape
    else:
        assert original_rgb_np_u8 is not None
        assert isinstance(original_rgb_np_u8, np.ndarray)
        assert original_rgb_np_u8.ndim == 3
        assert original_rgb_np_u8.dtype == np.uint8
        assert original_rgb_np_u8.shape[2] == 3
        photograph_height_in_pixels, photograph_width_in_pixels, _ = original_rgb_np_u8.shape

    for ad_placement_descriptor in textured_ad_placement_descriptors:
        assert isinstance(ad_placement_descriptor, AdPlacementDescriptor)
        texture_rgba_np_f32 = ad_placement_descriptor.texture_rgba_np_f32
        assert isinstance(texture_rgba_np_f32, np.ndarray)
        assert texture_rgba_np_f32.ndim == 3
        assert texture_rgba_np_f32.dtype == np.float32, f"texture_rgba_np_f32 has dtype {texture_rgba_np_f32.dtype} but it should be np.float32"
        assert texture_rgba_np_f32.shape[2] == 4, f"texture_rgba_np_f32.shape[2] is {texture_rgba_np_f32.shape[2]} but it should be 4"
        if use_linear_light:
            the_min = np.min(texture_rgba_np_f32)
            the_max = np.max(texture_rgba_np_f32)
            assert the_min >= -0.1, f"texture_rgba_np_f32 has a min of {the_min} but it should be >= 0.0"
            assert the_max <= 1.1, f"texture_rgba_np_f32 has a max of {the_max} but it should be <= 1.0"
            
    assert isinstance(camera_pose, CameraParameters)

    # the final step will be placing the "transparency film" on top:
    if anti_aliasing_factor == 2:
        upscaled_mask_hw_np_u8 = mask_hw_np_u8.repeat(2, axis=0).repeat(2, axis=1)
    elif anti_aliasing_factor == 3:
        upscaled_mask_hw_np_u8 = mask_hw_np_u8.repeat(3, axis=0).repeat(3, axis=1)
    else:
        upscaled_mask_hw_np_u8 = mask_hw_np_u8
    
    # this is the largest collection of pixels that might be affected, considering it is to be masked:
    ijs = np.argwhere(upscaled_mask_hw_np_u8 < 255)
    
    # if speed is not a concern, you could use all the pixels like this:
    # ijs = np.argwhere(upscaled_mask_hw_np_u8 < 256)
   

    # make an RGBA 3D rendering of all the quads called rendering_of_quads_rgba_hwc_np_f32:
    rgba_values_at_those_ijs = render_ads_on_subregion(
        ad_placement_descriptors=textured_ad_placement_descriptors,
        ijs=ijs,
        photograph_width_in_pixels=anti_aliasing_factor * photograph_width_in_pixels,  # needed to convert ijs to normalized [-1,1] x [9/16, 9/16] normalized coordinates
        photograph_height_in_pixels=anti_aliasing_factor* photograph_height_in_pixels,
        camera_parameters=camera_pose,
    )
    
    assert rgba_values_at_those_ijs.shape == (ijs.shape[0], 4)
    assert rgba_values_at_those_ijs.dtype == np.float32

    # place them in 2D:
    rendering_of_quads_rgba_hwc_np_f32 = np.zeros(
        shape=(
            photograph_height_in_pixels * anti_aliasing_factor,
            photograph_width_in_pixels * anti_aliasing_factor,
            4
        ),
        dtype=np.float32
    )

    rendering_of_quads_rgba_hwc_np_f32[ijs[:, 0], ijs[:, 1], :] =  rgba_values_at_those_ijs
    
    # prii_linear_f32(
    #     rendering_of_quads_rgba_hwc_np_f32,
    #     caption="rendering_of_quads_rgba_hwc_np_f32 inside insert_quads_into_camera_posed_image_behind_mask"
    # )
    
    if anti_aliasing_factor == 1:
        anti_aliased_f32 = rendering_of_quads_rgba_hwc_np_f32[:, :, :3]
    elif anti_aliasing_factor == 2:
        anti_aliased_f32 = (
            rendering_of_quads_rgba_hwc_np_f32[::2, ::2, :3]
            + rendering_of_quads_rgba_hwc_np_f32[1::2, ::2, :3]
            + rendering_of_quads_rgba_hwc_np_f32[::2, 1::2, :3]
            + rendering_of_quads_rgba_hwc_np_f32[1::2, 1::2, :3]
        ) / 4.0
    elif anti_aliasing_factor == 3:
        anti_aliased_f32 = (
            rendering_of_quads_rgba_hwc_np_f32[::3, ::3, :3]
            + rendering_of_quads_rgba_hwc_np_f32[1::3, ::3, :3]
            + rendering_of_quads_rgba_hwc_np_f32[2::3, ::3, :3]
            + rendering_of_quads_rgba_hwc_np_f32[::3, 1::3, :3]
            + rendering_of_quads_rgba_hwc_np_f32[1::3, 1::3, :3]
            + rendering_of_quads_rgba_hwc_np_f32[2::3, 1::3, :3]
            + rendering_of_quads_rgba_hwc_np_f32[::3, 2::3, :3]
            + rendering_of_quads_rgba_hwc_np_f32[1::3, 2::3, :3]
            + rendering_of_quads_rgba_hwc_np_f32[2::3, 2::3, :3]
        ) / 9.0
    else:
        raise Exception(f"anti_aliasing_factor={anti_aliasing_factor} is not supported")

    # Do the composition:
    if use_linear_light:
        # prii_linear_f32(anti_aliased_f32, caption="anti_aliased_f32 inside insert_quads_into_camera_posed_image_behind_mask")

        mask_rgba_np_linear_f32 = np.zeros(
            shape=(
                mask_hw_np_u8.shape[0],
                mask_hw_np_u8.shape[1],
                4
            ),
            dtype=np.float32
        )
        mask_rgba_np_linear_f32[:, :, :3] = original_rgb_np_linear_f32
        mask_rgba_np_linear_f32[:, :, 3] = mask_hw_np_u8 / 255.0
        
        composition_rgb_np_linear_f32 = feathered_paste_for_images_of_the_same_size_linear_f32(
            bottom_layer_rgb_np_linear_f32=anti_aliased_f32,
            top_layer_rgba_np_linear_f32=mask_rgba_np_linear_f32
        )
        assert composition_rgb_np_linear_f32.shape == (photograph_height_in_pixels, photograph_width_in_pixels, 3)
        # prii_linear_f32(
        #     composition_rgb_np_linear_f32,
        #     caption="composition_rgba_np_linear_f32 inside insert_quads_into_camera_posed_image_behind_mask"
        # )
        
        return composition_rgb_np_linear_f32
    else:
        mask_rgba_np_u8 = np.zeros(
            shape=(
                mask_hw_np_u8.shape[0],
                mask_hw_np_u8.shape[1],
                4
            ),
            dtype=np.uint8
        )
        mask_rgba_np_u8[:, :, :3] = original_rgb_np_u8
        mask_rgba_np_u8[:, :, 3] = mask_hw_np_u8
    
        anti_aliased_u8 = np.round(anti_aliased_f32 * 255).clip(0, 255).astype(np.uint8)
    
        composition_rgb_np_u8 = feathered_paste_for_images_of_the_same_size(
            bottom_layer_color_np_uint8=anti_aliased_u8,
            top_layer_rgba_np_uint8=mask_rgba_np_u8,
        )
        assert composition_rgb_np_u8.shape == (photograph_height_in_pixels, photograph_width_in_pixels, 3)
        # prii(composition_rgba_np_uint8, caption="composition_rgba_np_uint8")
        
        return composition_rgb_np_u8

