"""
Opening image files has a lot more nuance that people are ready to admit. It defines:
* open_as_hwc_rgb_np_uint8
* open_as_hwc_rgba_np_uint8
"""
import PIL
import PIL.Image
import numpy as np
# import imageio as iio
from pathlib import Path

# define what from image_openers import * should import:
__all__ = [
    "open_as_hwc_rgb_np_uint8",
    "open_as_hwc_rgba_np_uint8",
    "open_as_grayscale_regardless",
    "open_a_grayscale_png_barfing_if_it_is_not_grayscale",
    "open_alpha_channel_image_as_a_single_channel_grayscale_image",
    "open_image_as_rgb_np_uint8_ignoring_any_alpha",
]

def open_as_hwc_rgb_np_uint8(image_path: Path):
    """
    Opens an image file path to be RGB np.uint8 H x W x C
    """
    pil = PIL.Image.open(str(image_path)).convert("RGB")
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 3
    return image_np_uint8


def open_as_hwc_rgba_np_uint8(image_path: Path):
    """
    Opens an image file path to be RGBA np.uint8 H x W x C=4
    regardless of whether the image is actually RGBA or not,
    i.e. it will add a fully opaque alpha channel if the image
    does not have one.
    """
    pil = PIL.Image.open(str(image_path)).convert("RGBA")
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 3
    assert image_np_uint8.shape[2] == 4
    return image_np_uint8


def open_as_grayscale_regardless(image_path):
    """
    There are images that are truly grayscale, and there are images that look a
    lot like grayscale, but are actually RGB(a) images where there are R, G, and B channels, and mybe an alpha channel.
    This function opens the image as grayscale regardless of whether it is truly grayscale or not.
    """
    pil = PIL.Image.open(str(image_path))
    if pil.mode == "L":
        gray_np_u8 = np.array(pil)
       
        return gray_np_u8
    else:
        image_np_uint8 = np.array(pil)
        assert image_np_uint8.ndim == 3
        assert image_np_uint8.dtype == np.uint8
        assert image_np_uint8.shape[2] == 3 or image_np_uint8.shape[2] == 4
        image_np_uint8 = image_np_uint8[:, :, :3]
        gray_np_u8 = np.mean(image_np_uint8, axis=2).astype(np.uint8)

    assert gray_np_u8.ndim == 2
    assert gray_np_u8.dtype == np.uint8
    return gray_np_u8


def open_a_grayscale_png_barfing_if_it_is_not_grayscale(image_path):
    """
    Opens a PNG image file path if and only if it is
    a grayscale (not RGB, not RGBA)
    """
    pil = PIL.Image.open(str(image_path))
    assert pil.mode == "L", (
        f"Barfing because the file {image_path} is not a grayscale PNG. "
        f"Confirm for yourself via:\n\n    exiftool {image_path}"
    )
    image_np_uint8 = np.array(pil)
    assert image_np_uint8.ndim == 2
    return image_np_uint8


def open_alpha_channel_image_as_a_single_channel_grayscale_image(abs_file_path):
    """
    Sometimes we store the alpha channel as the 4th channel of an RGBA PNG,
    and sometimes we store the alpha as grayscale image.
    This function handles both cases.
    """
    attempt = np.asarray(PIL.Image.open(abs_file_path))
    if attempt.ndim == 3:
        assert (
            attempt.shape[2] == 4
        ), f"{abs_file_path} need to either have 4 channels with the 4th channel being considered the alpha channel, or just one channel, and then that one channel is considered the alpha channel."
        ans = attempt[:, :, 3].copy()
    elif attempt.ndim == 2:
        ans = attempt
    else:
        raise Exception(f"ERROR: I don't know how to handle this file {abs_file_path}")
    return ans



def open_image_as_rgb_np_uint8_ignoring_any_alpha(abs_file_path):
    """
    Opens an image that contains color for the color part,
    ignoring any alpha
    """
    attempt = np.asarray(PIL.Image.open(abs_file_path))
    assert attempt.ndim == 3
    assert (
        attempt.shape[2] == 3 or attempt.shape[2] == 4
    ), f"{abs_file_path} need to either have 3 or 4 channels to get out the rgb part"
    ans = attempt[:, :, :3].copy()
    return ans