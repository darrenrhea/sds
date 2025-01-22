import PIL
import PIL.Image
import numpy as np
from pathlib import Path


def open_alpha_channel_image_as_a_single_channel_grayscale_image(
    abs_file_path: Path
):
    """
    open_alpha_channel_image_as_a_single_channel_grayscale_image
    
    Sometimes we store the alpha channel as the 4th channel of an RGBA PNG,
    and sometimes we store the alpha as 1-channel grayscale image.
    This function opens just the alpha channel in both cases.
    """
    attempt = np.array(PIL.Image.open(abs_file_path))
    if attempt.ndim == 3:
        assert (
            attempt.shape[2] == 4
        ), f"{abs_file_path} need to either have 4 channels with the 4th channel being considered the alpha channel, or just one channel, and then that one channel is considered the alpha channel."
        ans = attempt[:, :, 3].copy()
    elif attempt.ndim == 2:
        ans = attempt
    else:
        raise Exception(
            f"ERROR: you sent open_alpha_channel_image_as_a_single_channel_grayscale_image this image:\n"
            f"{abs_file_path}\n"
            f"which has {attempt.shape[2]} channels.\n"
        )
    return ans
