from forward_project_world_points import (
     forward_project_world_points
)
import numpy as np
from scipy.ndimage import map_coordinates
from CameraParameters import CameraParameters


def rip_world_points(
    rgb_hwc_np_u8: np.ndarray,
    camera_pose: CameraParameters,
    xyzs: np.ndarray, # a list of world locations you want to convert to pixel locations
) -> np.ndarray:
    """
    Return the rgba values at the 3D world locations xyzs
    in the given rgb photograph rgb_hwc_np_u8
    that correspond to the to the world points in xyzs.
    The alpha channel of the rgba values will be 255 unless it is off-the-map,
    at which point it will be zero, i.e. the alpha channel is a visibility_mask.
    """
    assert isinstance(rgb_hwc_np_u8, np.ndarray)
    assert rgb_hwc_np_u8.ndim == 3
    assert rgb_hwc_np_u8.shape[2] == 3, "rgb_hwc_np_u8 should have three channels"
    assert rgb_hwc_np_u8.dtype == np.uint8

    photograph_height_in_pixels = rgb_hwc_np_u8.shape[0]
    photograph_width_in_pixels = rgb_hwc_np_u8.shape[1]

    rgba_hwc_np_u8 = np.concatenate(
        [
            rgb_hwc_np_u8,
            np.full(
                shape=(photograph_height_in_pixels, photograph_width_in_pixels, 1),
                fill_value=255,
                dtype=np.uint8
            )
        ],
        axis=2
    )

    rgba_hwc_np_f32 = rgba_hwc_np_u8.astype(np.float32)

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

    # print(f"{ijs.shape=}")
    
    rgba_values_f32 = np.zeros(
        shape=(xyzs.shape[0], 4),
        dtype=np.float32
    )

    for c in range(0, 4):
        rgba_values_f32[:, c] = map_coordinates(
            input=rgba_hwc_np_f32[:, :, c],
            coordinates=ijs.T,
            order=1,
            cval=0,
        )
    
    assert rgba_values_f32.shape[1] == 4

    return rgba_values_f32
