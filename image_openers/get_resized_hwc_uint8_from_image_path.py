import PIL
import PIL.Image
from pathlib import Path
import numpy as np



def get_resized_hwc_uint8_from_image_path(image_path, width, height):
    """
    given a string or Path where an image file with the correct extension is stored,
    returns a 3 tuple, which contains:
    
    hwc, i.e. height x width x rgb channels, uint8 numpy array of that image,
    as well as
    the original_image_width
    and
    the original_image_height.
    Any alpha channel will be dropped if present.
    Here is a typical invokation:

    ::

        hwc_uint8, original_height, original_width = get_resized_hwc_uint8_from_image_path(
            image_path="cat.jpg",
            height=224,
            width=224,
        )

        assert hwc_uint8.shape == (224, 224, 3)
        assert hwc_uint8.dtype == np.uint8
    
    """
    color_pil_image = PIL.Image.open(str(Path(image_path).expanduser())).convert("RGB")

    original_image_width, original_image_height = color_pil_image.size
    resized_pil_image = color_pil_image.resize(
        size=(width, height), resample=PIL.Image.BICUBIC, box=None
    )
    np_hwc_uint8 = np.array(resized_pil_image)
    return np_hwc_uint8, original_image_width, original_image_height

