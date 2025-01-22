from get_width_and_height_that_fits_on_the_screen import (
     get_width_and_height_that_fits_on_the_screen
)
from get_click_on_image import (
     get_click_on_image
)
from typing import Optional, Tuple
import numpy as np
import cv2


def get_click_on_resized_image(
    max_display_width: int,
    max_display_height: int,
    rgb_hwc_np_u8: Optional[np.ndarray] = None,
    instructions_string: str = None,
) -> Optional[Tuple[float, float]]:
    """
    Given an image that may be:
    1. too small to accurately click, or
    2. too big to fit on the screen,
    this resizes it, then gets a maybe click on it.

    Given an image that may be two big to fit on the screen,
    or too small to accurately click on (for instance the window title gets in the way)
    this resizes it, then gets a maybe click / optional click on it,
    i.e. None or an (x, y) coordinate tuple, y growing down, x growing right.
    """
    height, width = rgb_hwc_np_u8.shape[:2]
    
    new_width, new_height = get_width_and_height_that_fits_on_the_screen(
        width=width,
        height=height,
        max_display_width=max_display_width,
        max_display_height=max_display_height,
    )

    resized = cv2.resize(
        rgb_hwc_np_u8,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )

    click = get_click_on_image(
        rgb_hwc_np_u8=resized,
        instructions_string=instructions_string
    )
    if click is None:
        return None
    else:
        x, y = click
        x = x * width / new_width
        y = y * height / new_height
        return [x, y]

   
