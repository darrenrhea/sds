from forward_project_world_points import (
     forward_project_world_points
)
import numpy as np
from scipy.ndimage import map_coordinates
from CameraParameters import CameraParameters


def rip_mask_values_at_these_world_points(
    mask_hw_np_u8: np.ndarray,
    camera_pose: CameraParameters,
    xyzs: np.ndarray, # a list of world locations you want to convert to pixel locations
) -> np.ndarray:
    """
    Return the mask values of the pixels in the photograph that correspond to the
    to to the world points in xyzs.
    """
    assert isinstance(mask_hw_np_u8, np.ndarray)
    assert mask_hw_np_u8.ndim == 2
    assert mask_hw_np_u8.dtype == np.uint8

    photograph_height_in_pixels = mask_hw_np_u8.shape[0]
    photograph_width_in_pixels = mask_hw_np_u8.shape[1]

    mask_hw_np_f32 = mask_hw_np_u8.astype(np.float32)

    ijs = forward_project_world_points(
        photograph_width_in_pixels=photograph_width_in_pixels,
        photograph_height_in_pixels=photograph_height_in_pixels,
        camera_pose=camera_pose,
        xyzs=xyzs,
    )

    # i_s = ijs[:, 0]
    # j_s = ijs[:, 1]
    
    # print(f"{np.min(i_s)=}")
    # print(f"{np.max(i_s)=}")
    # print(f"{np.min(j_s)=}")
    # print(f"{np.max(j_s)=}")
     
    mask_values_f32 = map_coordinates(
        input=mask_hw_np_f32,
        coordinates=ijs.T,
        order=1,
        cval=0,
    )
    return mask_values_f32
