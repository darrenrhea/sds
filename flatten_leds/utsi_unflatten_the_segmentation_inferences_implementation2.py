from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from perspective_insert_mask import (
     perspective_insert_mask
)
import numpy as np
from CameraParameters import (
     CameraParameters
)

def utsi_unflatten_the_segmentation_inferences_implementation2(
    flat_mask_hw_np_u8: np.ndarray,  # this gets unflattened
    camera_pose: CameraParameters,
    ad_origin,
    u,
    v,
    ad_height,
    ad_width,
    rip_height: int,
    rip_width: int,
    photograph_height_in_pixels: int,
    photograph_width_in_pixels: int,
):
    """
    Suppose for a given video frame,
    For each LED board, you have flattened the RGB then inferred in flattened space.
    This puts those inferences back into the original 3D perspective.
    
    Having inferred on flattened LED boards,
    you need to unflatten it back into a perspectivized segmentation.
    
    TODO: multiple LED boards unioned.
    """
    assert isinstance(camera_pose, CameraParameters)
    assert flat_mask_hw_np_u8.shape == (rip_height, rip_width)

    ad_placement_descriptor = AdPlacementDescriptor(
        name="mirror_world_floor",  # so far only one LED board in NBA
        origin=ad_origin,
        u=u,
        v=v,
        height=ad_height,
        width=ad_width,
    )
    
    ad_placement_descriptors = [
        ad_placement_descriptor,
    ]
    
    mask_hw_np_u8 = perspective_insert_mask(
        flat_mask_hw_np_u8=flat_mask_hw_np_u8,
        ad_placement_descriptors=ad_placement_descriptors,
        camera_pose=camera_pose,
        photograph_height_in_pixels=photograph_height_in_pixels,
        photograph_width_in_pixels=photograph_width_in_pixels,
    )
  
    return mask_hw_np_u8

