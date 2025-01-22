from scipy.ndimage.filters import gaussian_filter
from convert_u8_to_linear_f32 import (
     convert_u8_to_linear_f32
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import PIL
import PIL.Image
import PIL.ImageFilter
import numpy as np


def get_rgba_hwc_np_f32_from_texture_id(
    texture_id: str,
    use_linear_light: bool,
    blur_radius: float = 0.0
):
    """
    TODO: Wow. This is doing to many things at once.
    might not it be simpler to just use sha256 directly?
    The resolution from the continuum id texture_id to sha256 should be
    done in a separate function.

    TODO: Also, the blurring part should be done in a separate function.

    It is understood that color channels have values in the range [0, 1].
    """
    texture_id_to_sha256 = dict(
        [
            (
                "different_here",
                "8c4b108c53092b7e3cf7e28add26a319f47e2ad87b02bf3a3715d2903acd3b31",
            ),
            (
                "23-24_BOS_CORE",
                "bc8802241b2b65425322d9ae25cb0791bc756e6d1b221d4d13879708bc23b6aa",
            ),
            (
                "24-25_HOU_CORE",
                "5870846017ac9cdd5bb4383fe8d6fa60e0eff0db6160453cf4ae1a481a3f65ef",
            )
        ]
    )
    sha256 = texture_id_to_sha256[texture_id]
    file_path = get_file_path_of_sha256(sha256=sha256)

    # this ensures that the image have 4 channels regardless of the original image:
    rgba_hwc_np_u8 = open_as_rgba_hwc_np_u8(
        image_path=file_path
    )

    if use_linear_light:                
        assert rgba_hwc_np_u8.shape[2] == 4

        rgba_hwc_np_linear_f32 = convert_u8_to_linear_f32(
            x=rgba_hwc_np_u8
        )
        if blur_radius == 0.0:
            return rgba_hwc_np_linear_f32
        else:
            blurred_rgb_linear_f32 = np.zeros(
                shape=(rgba_hwc_np_linear_f32.shape[0], rgba_hwc_np_linear_f32.shape[1], 3),
                dtype=np.float32,
            )

            for i in range(3):
                blurred_rgb_linear_f32[:, :, i] = gaussian_filter(
                    rgba_hwc_np_linear_f32[:, :, i],
                    sigma=blur_radius,
                )
 
            blurred_rgba_hwc_np_linear_f32 = np.dstack(
                (
                    blurred_rgb_linear_f32,
                    rgba_hwc_np_linear_f32[:, :, 3]
                )
            )

            return blurred_rgba_hwc_np_linear_f32
    else:
        image_pil = PIL.Image.fromarray(rgba_hwc_np_u8)

        blur_filter = PIL.ImageFilter.GaussianBlur(
            radius=blur_radius
        )

        blurred_pil = image_pil.filter(filter=blur_filter)
    
        blurred_rgba_hwc_u8 = np.array(blurred_pil)

        rgba_hwc_np_f32 = blurred_rgba_hwc_u8.astype("float32")
        
        rgba_hwc_np_f32 /= 255.0
        return rgba_hwc_np_f32
