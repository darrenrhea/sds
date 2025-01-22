from typing import Tuple
import numpy as np


def get_width_and_height_that_fits_on_the_screen(
    width, height, max_display_width, max_display_height
) -> Tuple[int, int]:
    """
    Sometimes you have a ridiculously large image, or a ridiculously small image,
    and you want to resize it to fit on the screen.

    Given the width and height of an image, and the maximum width and maximum_height
    that you can fit, this gives back a new width and new_height that
    approximately preserves the aspect ratio of the original image
    while fitting within the max_display_width and max_display_height.
    """
    # how much do we have to shrink the image to fit it within max_display_width:
    width_shrink_factor = width / max_display_width

    # how much do we have to shrink the image to fit it within max_display_height:
    height_shrink_factor = height / max_display_height

    shrink_factor = max(width_shrink_factor, height_shrink_factor)

    # the new width and height:
    new_width = int(np.round(width / shrink_factor))
    new_height = int(np.round(height / shrink_factor))

    assert new_width <= max_display_width
    assert new_height <= max_display_height
    assert new_width == max_display_width or new_height == max_display_height

    return new_width, new_height


