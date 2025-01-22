import numpy as np
from Drawable2DImage import Drawable2DImage


def draw_points_on_rasterized_image(
    original_rgb_np_u8: np.ndarray,  # does not mutate this
    xys: np.ndarray,
) -> np.ndarray:
    """
    Suppose you have a rasterized image and some points in pixel coordinates
    where x is left to right from 0 to width - 1 and y is top to bottom from 0 to height - 1.
    and you just want to draw them on the image.
    """
    assert isinstance(xys, np.ndarray)
    assert xys.ndim == 2
    assert xys.shape[1] == 2

    assert isinstance(original_rgb_np_u8, np.ndarray)
    assert original_rgb_np_u8.dtype == np.uint8
    assert original_rgb_np_u8.ndim == 3
    assert original_rgb_np_u8.shape[2] == 3

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

    for xy in xys:
       
        x_pixel = xy[0]
        y_pixel = xy[1]
        
        drawable_image.draw_plus_at_2d_point(
            x_pixel=x_pixel,
            y_pixel=y_pixel,
            rgb=(0, 255, 0),
            size=3,
            text=""
        )
    
    return np.array(drawable_image.image_pil)
