from typing import Optional
from nuke_lens_distortion import nuke_world_to_pixel_coordinates
import numpy as np
from Drawable2DImage import Drawable2DImage
from CameraParameters import CameraParameters


def draw_named_3d_points(
    original_rgb_np_u8: np.ndarray,  # does not mutate this
    camera_pose: Optional[CameraParameters],
    landmark_name_to_xyz: dict,
) -> np.ndarray:
    """
    You have a video frame and a camera pose. You have a bunch of 3D points in the world. You want to draw them on the image.
    """
    if camera_pose is None:
        pass
    else:
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

    if camera_pose is not None:
        pixel_points = dict()
        for landmark_name, p_giwc in landmark_name_to_xyz.items():
            p_giwc = landmark_name_to_xyz[landmark_name]
        

            x_pixel, y_pixel, is_visible = nuke_world_to_pixel_coordinates(
                p_giwc=np.array(p_giwc),
                camera_parameters=camera_pose,
                photograph_width_in_pixels=photograph_width_in_pixels,
                photograph_height_in_pixels=photograph_height_in_pixels
            )
            
            if is_visible:
                # print(f"    {landmark_name} visible at {x_pixel}, {y_pixel}")
                pixel_points[landmark_name] = dict(
                    i=int(y_pixel),
                    j=int(x_pixel),
                )
            else:
                pass
                # print(f"    {landmark_name} is not visible.")

    

        for landmark_name, dct in pixel_points.items():
            drawable_image.draw_plus_at_2d_point(
                x_pixel=dct["j"],
                y_pixel=dct["i"],
                rgb=(0, 255, 0),
                size=10,
                text=landmark_name
            )
        
    return np.array(drawable_image.image_pil)
