from get_click_on_resized_image import (
     get_click_on_resized_image
)
from typing import Optional, Tuple
import numpy as np


def get_click_on_subrectangle_of_image(
    j_min: int,
    j_max: int,
    i_min: int,
    i_max: int,
    rgb_hwc_np_u8: Optional[np.ndarray],
    max_display_width: int,
    max_display_height: int,
    instructions_string: str = None
) -> Optional[Tuple[float, float]]:
    """
    Sometimes we want to zoom-in on a subrectangle of an image so that we can see what we are doing,
    then click on a point.
    """

    assert i_min < i_max
    assert j_min < j_max
    assert i_min >= 0
    assert j_min >= 0
    assert i_max <= rgb_hwc_np_u8.shape[0]
    assert j_max <= rgb_hwc_np_u8.shape[1]

    subrectangle = rgb_hwc_np_u8[i_min:i_max, j_min:j_max, :]

    click = get_click_on_resized_image(
        max_display_width=max_display_width,
        max_display_height=max_display_height,
        rgb_hwc_np_u8=subrectangle,
        instructions_string=instructions_string
    )
    if click is None:
        return None
    else:
        x, y = click
        x = x + j_min
        y = y + i_min
        return [x, y]

   
