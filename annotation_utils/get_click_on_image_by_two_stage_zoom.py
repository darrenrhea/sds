from typing import Optional, Tuple
from get_click_on_subrectangle_of_image import (
     get_click_on_subrectangle_of_image
)
from get_click_on_resized_image import (
     get_click_on_resized_image
)
import numpy as np


def get_click_on_image_by_two_stage_zoom(
    max_display_width: int,
    max_display_height: int,
    rgb_hwc_np_u8: np.ndarray,
    instructions_string: str = None,
) -> Optional[Tuple[float, float]]:
    """
    Given an image that may be:
    1. too big to fit on the screen, or
    2. too small to accurately click on,
    this resizes it, then gets a maybe click on it.
    """
    
    click = get_click_on_resized_image(
        max_display_width=max_display_width,
        max_display_height=max_display_height,
        rgb_hwc_np_u8=rgb_hwc_np_u8,
        instructions_string=instructions_string
    )

    if click is None:
        return None
    else:
        x_int = int(click[0])
        y_int = int(click[1])
        i_min = max(0, y_int - 50)
        i_max = min(rgb_hwc_np_u8.shape[0], y_int + 50)
        j_min = max(0, x_int - 50)
        j_max = min(rgb_hwc_np_u8.shape[1], x_int + 50)
        click = get_click_on_subrectangle_of_image(
            j_min=j_min,
            j_max=j_max,
            i_min=i_min,
            i_max=i_max,
            rgb_hwc_np_u8=rgb_hwc_np_u8,
            instructions_string=instructions_string,
            max_display_width=max_display_width,
            max_display_height=max_display_height,
        )

        if click is None:
            return None
        else:
            return click