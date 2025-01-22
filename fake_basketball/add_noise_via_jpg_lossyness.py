from io import BytesIO

import PIL.Image
import numpy as np


def add_noise_via_jpg_lossyness(
    rgb_hwc_np_u8: np.ndarray,
    jpeg_quality: int
):
    """
    Add noise to an rgb image by "saving it" (this happens in RAM, not on disk)
    as a low jpeg_quality JPEG and then loading it back.
    """
    assert isinstance(rgb_hwc_np_u8, np.ndarray)
    assert rgb_hwc_np_u8.dtype == np.uint8
    assert rgb_hwc_np_u8.ndim == 3, f"ERROR you sent add_noise_via_jpg_lossyness an image with shape {rgb_hwc_np_u8.shape}"
    assert rgb_hwc_np_u8.shape[2] == 3, f"ERROR you sent add_noise_via_jpg_lossyness an image with shape {rgb_hwc_np_u8.shape}"
    
    assert 1 <= jpeg_quality
    assert jpeg_quality <= 95

    image_pil = PIL.Image.fromarray(rgb_hwc_np_u8)
    membuf = BytesIO()

    image_pil.save(
        fp=membuf,
        format="JPEG",
        subsampling=1,  # i.e. 4:2:2
        jpeg_quality=jpeg_quality
    )

    membuf.seek(0)
    image_pil_noisy = PIL.Image.open(membuf)
    rgb_hwc_noisy_np_u8 = np.array(image_pil_noisy)

    return rgb_hwc_noisy_np_u8
