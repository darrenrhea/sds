from render_masks_on_subregion import (
     render_masks_on_subregion
)
from typing import List
import numpy as np
import PIL
import PIL.Image
from CameraParameters import CameraParameters


def perspective_insert_mask(
    camera_pose: CameraParameters,
    flat_mask_hw_np_u8: np.ndarray,
    # flat_mask_hw_np_u8 is a flat mask, e.g. the result of segmenting a flattened LED board
    ad_placement_descriptors: List,
    photograph_width_in_pixels: int,
    photograph_height_in_pixels: int
) -> np.ndarray:
    """
    If you have a "flattened mask", e.g. a segmentation infererence of the flattened LED board,
    you might want to re-perspectivize it into 3 dimensional perspective picture.
    """
    assert isinstance(flat_mask_hw_np_u8, np.ndarray)
    assert flat_mask_hw_np_u8.ndim == 2
    assert flat_mask_hw_np_u8.dtype == np.uint8


    texture_hw_np_f32 = flat_mask_hw_np_u8.astype(np.float32)
    assert texture_hw_np_f32.dtype == np.float32

    anti_aliasing_factor = 1
   
    unflat_mask_hw_np_u8 = np.zeros(
        (
            photograph_height_in_pixels,
            photograph_width_in_pixels,
        ),
        dtype=np.uint8
    )
    # this is a strange way to say everywhere:
    # if we had an estimate it can go much faster.
    ijs = np.argwhere(unflat_mask_hw_np_u8[:, :] == 0)

    for ad_placement_descriptor in ad_placement_descriptors:
        ad_placement_descriptor.texture_hw_np_f32 = texture_hw_np_f32
    
    mask_values_at_those_ijs = render_masks_on_subregion(
        ad_placement_descriptors=ad_placement_descriptors,
        ijs=ijs,
        photograph_width_in_pixels=photograph_width_in_pixels,  # needed to convert ijs to normalized [-1,1] x [9/16, 9/16] normalized coordinates
        photograph_height_in_pixels=photograph_height_in_pixels,
        camera_parameters=camera_pose,
    )

    # place them in 2D:
    ad_placement_accumulator = np.zeros(
        (
            photograph_height_in_pixels * anti_aliasing_factor,
            photograph_width_in_pixels * anti_aliasing_factor,
        ),
        dtype=np.uint8
    ) + 255

    ad_placement_accumulator[ijs[:, 0], ijs[:, 1]] =  mask_values_at_those_ijs
    
   

    final_pil = PIL.Image.fromarray(ad_placement_accumulator, mode="L")
    antialiased_pil = final_pil.resize(
        (photograph_width_in_pixels, photograph_height_in_pixels),
        resample=PIL.Image.Resampling.BILINEAR
    )
    perspectivized_mask_np_u8 = np.array(antialiased_pil)
    
    return perspectivized_mask_np_u8

