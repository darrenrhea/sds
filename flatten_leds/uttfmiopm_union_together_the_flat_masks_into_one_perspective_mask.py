from typing import List
from utsi_unflatten_the_segmentation_inferences_implementation2 import (
     utsi_unflatten_the_segmentation_inferences_implementation2
)

from CameraParameters import (
     CameraParameters
)

def uttfmiopm_union_together_the_flat_masks_into_one_perspective_mask(
    pairs_of_flat_mask_and_ad_descriptor: List,
    camera_pose: CameraParameters,
    photograph_height_in_pixels: int,
    photograph_width_in_pixels: int,
):
    """
    After we have segment each flattened ad board separately,
    we may want to union them together into one perspective mask.
    TODO: different boards can have different rip_heights and rip_widths
    TODO: multiple boards is not actually implemented yet -- should be an accumulation of the masks.
    """
    assert isinstance(camera_pose, CameraParameters)
    # TODO: this should be per ad board:
    rip_height = 256
    rip_width = 4268

    
   
   
    assert len(pairs_of_flat_mask_and_ad_descriptor) == 1, "Not implemented for more than one ad placement descriptor"
    flat_mask_hw_np_u8, ad_placement_descriptor = pairs_of_flat_mask_and_ad_descriptor[0]

    ad_origin = ad_placement_descriptor.origin
    u = ad_placement_descriptor.u
    v = ad_placement_descriptor.v
    ad_height = ad_placement_descriptor.height
    ad_width = ad_placement_descriptor.width


    mask_hw_np_u8 = utsi_unflatten_the_segmentation_inferences_implementation2(
        flat_mask_hw_np_u8=flat_mask_hw_np_u8,
        camera_pose=camera_pose,
        rip_height=rip_height,
        rip_width=rip_width,
        ad_origin=ad_origin,
        u=u,
        v=v,
        ad_height=ad_height,
        ad_width=ad_width,
        photograph_height_in_pixels=photograph_height_in_pixels,
        photograph_width_in_pixels=photograph_width_in_pixels,
    )

    return mask_hw_np_u8




