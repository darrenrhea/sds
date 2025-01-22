from nuke_lens_distortion import nuke_world_to_pixel_coordinates
import numpy as np
from Drawable2DImage import Drawable2DImage
from CameraParameters import CameraParameters


def draw_3d_points(
    original_rgb_np_u8: np.ndarray,  # does not mutate this
    camera_pose: CameraParameters,
    xyzs: np.ndarray,
) -> np.ndarray:
    """
    Suppose you have a photograph (usually a video frame) and a camera pose fit to it.
    You have a bunch of 3D points in the world. You want to draw them on the image.
    """
    assert isinstance(xyzs, np.ndarray)
    assert xyzs.ndim == 2
    assert xyzs.shape[1] == 3

    assert isinstance(camera_pose, CameraParameters)
    assert isinstance(original_rgb_np_u8, np.ndarray)
    assert original_rgb_np_u8.dtype == np.uint8
    assert original_rgb_np_u8.ndim == 3
    assert original_rgb_np_u8.shape[2] == 3

    photograph_width_in_pixels = original_rgb_np_u8.shape[1]
    photograph_height_in_pixels = original_rgb_np_u8.shape[0]

    original_rgba_np_u8 = np.zeros(
        (original_rgb_np_u8.shape[0], original_rgb_np_u8.shape[1], 4),
        dtype=np.uint8
    )
    original_rgba_np_u8[:, :, :3] = original_rgb_np_u8
    original_rgba_np_u8[:, :, 3] = 255

    drawable_image = Drawable2DImage(
        rgba_np_uint8=original_rgba_np_u8,
        expand_by_factor=2
    )

    for p_giwc in xyzs:
       
        x_pixel, y_pixel, is_visible = nuke_world_to_pixel_coordinates(
            p_giwc=np.array(p_giwc),
            camera_parameters=camera_pose,
            photograph_width_in_pixels=photograph_width_in_pixels,
            photograph_height_in_pixels=photograph_height_in_pixels
        )
        
        if is_visible:
            drawable_image.draw_plus_at_2d_point(
                x_pixel=x_pixel,
                y_pixel=y_pixel,
                rgb=(0, 255, 0),
                size=3,
                text=""
            )
        
    return np.array(drawable_image.image_pil)
