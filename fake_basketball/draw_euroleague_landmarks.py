
from get_euroleague_geometry import (
     get_euroleague_geometry
)
from prii import (
     prii
)
from nuke_lens_distortion import nuke_world_to_pixel_coordinates
import numpy as np
from Drawable2DImage import Drawable2DImage
from CameraParameters import CameraParameters


def draw_euroleague_landmarks(
    original_rgb_np_u8: np.ndarray,  # does not mutate this
    camera_pose: CameraParameters
):
    """
    This has become too specific now that we are doing NBA.
    Sometimes we are suspicious of the camera pose, so we draw the landmarks to see if they are in the right place.
    """
    
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

    

    geometry = get_euroleague_geometry()
    points = geometry["points"]
    landmark_names = [key for key in points.keys()]

    pixel_points = dict()

    for landmark_name in landmark_names:
        p_giwc = geometry["points"][landmark_name]
    

        x_pixel, y_pixel, is_visible = nuke_world_to_pixel_coordinates(
            p_giwc=np.array(p_giwc),
            camera_parameters=camera_pose,
            photograph_width_in_pixels=photograph_width_in_pixels,
            photograph_height_in_pixels=photograph_height_in_pixels
        )
        
        if is_visible:
            print(f"    {landmark_name} visible at {x_pixel}, {y_pixel}")
            pixel_points[landmark_name] = dict(
                i=int(y_pixel),
                j=int(x_pixel),
            )
        else:
            pass
            print(f"    {landmark_name} is not visible.")

   

    for landmark_name, dct in pixel_points.items():
        drawable_image.draw_plus_at_2d_point(
            x_pixel=dct["j"],
            y_pixel=dct["i"],
            rgb=(0, 255, 0),
            size=10,
            text=landmark_name
        )
    
    prii(drawable_image.image_pil)
