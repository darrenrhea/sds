import PIL.Image
import numpy as np
from typing import Tuple

def get_to_image_pil_from_one_of_these(
    grayscale_np_uint8=None,
    binary_np_uint8=None,
    rgb_np_uint8=None,
    rgb_chw_np_uint8=None,
    rgba_chw_np_uint8=None,
    rgba_np_uint8=None,
    bgra_np_u8=None,
    bgr_np_u8=None,
    image_pil=None,
) -> Tuple[PIL.Image.Image, int]:
    """
    Could put image_path into here as well.
    """
    sum = (
        int(grayscale_np_uint8 is not None) +
        int(binary_np_uint8 is not None) +
        int(rgb_np_uint8 is not None) +
        int(rgb_chw_np_uint8 is not None) +
        int(rgba_chw_np_uint8 is not None) +
        int(rgba_np_uint8 is not None) +
        int(image_pil is not None) +
        int(bgra_np_u8 is not None) +
        int(bgr_np_u8 is not None)
    )
    assert sum == 1

    if image_pil is not None:
        assert (
            isinstance(image_pil, PIL.Image.Image)
        ), f"image_pil must be a PIL.Image.Image but it is of type {type(image_pil)}"
        num_channels = 4 if image_pil.mode == "RGBA" else 3  # TODO: this is almost certainly wrong
        return image_pil, num_channels
    
    if bgr_np_u8 is not None:
        assert isinstance(bgr_np_u8, np.ndarray), f"bgr_np_u8 must be a np.ndarray but it is of type {type(bgr_np_u8)}"
        assert bgr_np_u8.ndim == 3, f"bgr_np_u8.ndim must be 3 but it is {bgr_np_u8.ndim=}"
        assert bgr_np_u8.dtype == np.uint8, f"bgr_np_u8.dtype must be np.uint8 but it is {bgr_np_u8.dtype=}"
        assert bgr_np_u8.shape[2] == 3, f"bgr_np_u8.shape[2] must be 3 but it is {bgr_np_u8.shape[2]=}"
        image_pil = PIL.Image.fromarray(bgr_np_u8[:, :, [2, 1, 0]]).convert("RGB")
        num_channels = 3
    
    if bgra_np_u8 is not None:
        assert isinstance(bgra_np_u8, np.ndarray), f"bgra_np_u8 must be a np.ndarray but it is of type {type(bgra_np_u8)}"
        assert bgra_np_u8.ndim == 3, f"bgra_np_u8.ndim must be 3 but it is {bgra_np_u8.ndim=}"
        assert bgra_np_u8.dtype == np.uint8, f"bgra_np_u8.dtype must be np.uint8 but it is {bgra_np_u8.dtype=}"
        assert bgra_np_u8.shape[2] == 4, f"bgra_np_u8.shape[2] must be 4 but it is {bgra_np_u8.shape[2]=}"
        image_pil = PIL.Image.fromarray(bgra_np_u8[:, :, [2, 1, 0, 3]]).convert("RGBA")
        num_channels = 4
    

    if rgba_np_uint8 is not None:
        assert isinstance(rgba_np_uint8, np.ndarray), f"rgba_np_uint8 must be a np.ndarray but it is of type {type(rgba_np_uint8)}"
        assert rgba_np_uint8.ndim == 3, f"rgba_np_uint8.ndim must be 3 but it is {rgba_np_uint8.ndim=}"
        assert rgba_np_uint8.dtype == np.uint8, f"rgba_np_uint8.dtype must be np.uint8 but it is {rgba_np_uint8.dtype=}"
        assert rgba_np_uint8.shape[2] == 4, f"rgba_np_uint8.shape[2] must be 4 but it is {rgba_np_uint8.shape[2]=}"
        image_pil = PIL.Image.fromarray(rgba_np_uint8).convert("RGBA")
        num_channels = 4
    
    if rgb_np_uint8 is not None:
        assert isinstance(rgb_np_uint8, np.ndarray), f"rgb_np_uint8 must be a np.ndarray but it is of type {type(rgb_np_uint8)}"
        assert rgb_np_uint8.ndim == 3, f"rgb_np_uint8.ndim must be 3 but it is {rgb_np_uint8.ndim=}"
        assert rgb_np_uint8.shape[2] == 3, f"rgb_np_uint8.shape[2] must be 3 but it is {rgb_np_uint8.shape[2]=}"
        assert rgb_np_uint8.dtype == np.uint8, f"rgb_np_uint8.dtype must be np.uint8 but it is {rgb_np_uint8.dtype=}"
        image_pil = PIL.Image.fromarray(rgb_np_uint8)
        num_channels = 3

    if rgb_chw_np_uint8 is not None:
        assert isinstance(rgb_chw_np_uint8, np.ndarray), f"rgb_chw_np_uint8 must be a np.ndarray but it is of type {type(rgb_chw_np_uint8)}"
        assert rgb_chw_np_uint8.ndim == 3, f"rgb_chw_np_uint8.ndim must be 3 but it is {rgb_chw_np_uint8.ndim=}"
        assert rgb_chw_np_uint8.shape[0] == 3, f"rgb_chw_np_uint8.shape[0] must be 3 but it is {rgb_chw_np_uint8.shape[0]=}"
        assert rgb_chw_np_uint8.dtype == np.uint8, f"rgb_chw_np_uint8.dtype must be np.uint8 but it is {rgb_chw_np_uint8.dtype=}"

        rgb_np_uint8 = np.transpose(rgb_chw_np_uint8, axes=(1, 2, 0))
        image_pil = PIL.Image.fromarray(rgb_np_uint8)
        num_channels = 3

    if rgba_chw_np_uint8 is not None:
        assert isinstance(rgba_chw_np_uint8, np.ndarray), f"rgba_chw_np_uint8 must be a np.ndarray but it is of type {type(rgba_chw_np_uint8)}"
        assert rgba_chw_np_uint8.ndim == 3, f"rgba_chw_np_uint8.ndim must be 3 but it is {rgba_chw_np_uint8.ndim=}"
        assert rgba_chw_np_uint8.shape[0] == 4, f"rgba_chw_np_uint8.shape[0] must be 4 but it is {rgba_chw_np_uint8.shape[0]=}"
        assert rgba_chw_np_uint8.dtype == np.uint8, f"rgba_chw_np_uint8.dtype must be np.uint8 but it is {rgba_chw_np_uint8.dtype=}"
        rgba_np_uint8 = np.transpose(rgba_chw_np_uint8, axes=(1, 2, 0))
        image_pil = PIL.Image.fromarray(rgb_np_uint8)
        num_channels = 4
    
    if binary_np_uint8 is not None:
        assert isinstance(binary_np_uint8, np.ndarray), f"binary_np_uint8 must be a np.ndarray but it is of type {type(binary_np_uint8)}"
        assert binary_np_uint8.ndim == 2, "binary_np_uint8.ndim must be 2"
        assert binary_np_uint8.dtype == np.uint8, "binary_np_uint8.dtype must be np.uint8"
        assert np.all(np.logical_or(binary_np_uint8 == 0, binary_np_uint8 == 1)), "Hey that isn't binary"
        np_uint8 = np.zeros(shape=binary_np_uint8.shape, dtype=np.uint8)
        np_uint8[binary_np_uint8 == 1] = 255
        image_pil = PIL.Image.fromarray(np_uint8)
        num_channels = 1

    if grayscale_np_uint8 is not None:
        if grayscale_np_uint8.dtype == bool:  # TODO: this is a bit of a hack
            grayscale_np_uint8 = grayscale_np_uint8.astype(np.uint8) * 255
        assert isinstance(grayscale_np_uint8, np.ndarray), f"grayscale_np_uint8 must be a np.ndarray but it is of type {type(grayscale_np_uint8)}"
        assert grayscale_np_uint8.ndim == 2, "grayscale_np_uint8.ndim must be 2"
        assert grayscale_np_uint8.dtype == np.uint8, f"grayscale_np_uint8.dtype must be np.uint8 but it is {grayscale_np_uint8.dtype=}"
        image_pil = PIL.Image.fromarray(grayscale_np_uint8)
        num_channels = 1

    assert isinstance(image_pil, PIL.Image.Image), f"image_pil must be a PIL.Image.Image but it is of type {type(image_pil)}"
    
    return image_pil, num_channels

