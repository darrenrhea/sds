from prii import (
     prii
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)

import better_json as bj
import subprocess

from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)

from get_a_temp_file_path import (
     get_a_temp_file_path
)

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


from make_rgb_hwc_np_u8_from_rgba_hwc_np_u8 import (
     make_rgb_hwc_np_u8_from_rgba_hwc_np_u8
)


def get_clicks_on_image(
    image_path: Optional[Path] = None,
    rgba_hwc_np_u8: Optional[np.ndarray] = None,
    instructions_string: str = None
) -> List[Tuple[int, int]]:
    """
    TODO: make this able to take in a large number of image representations
    a la prii.

    Get 0 or more clicks on the image, until the user presses spacebar.

    - image_path: Path to the image.
    - hwc_rgb_np_u8: HWC RGB numpy array of uint8.

    Returns:
    - clicks: List of clicks as (x, y)
    """
    assert image_path is None or rgba_hwc_np_u8 is None, "image_path and rgba_hwc_np_u8 cannot both be given."
    assert image_path is not None or rgba_hwc_np_u8 is not None, "one of image_path and rgba_hwc_np_u8 must be given."

    executable_path = Path(
        "~/init/bin/finitely_many_clicks_on_one_image.exe"
    ).expanduser().resolve()

    assert executable_path.is_file(), f"executable_path not found. {executable_path} Did you install annotation_app?"

    if image_path is not None:
        assert image_path.is_file(), f"image_path not found. {image_path}"
        rgba_hwc_np_u8 = open_as_rgba_hwc_np_u8(image_path)
    
    height = rgba_hwc_np_u8.shape[0]
    width = rgba_hwc_np_u8.shape[1]

    rgb_hwc_np_u8 = make_rgb_hwc_np_u8_from_rgba_hwc_np_u8(
        rgba_hwc_np_u8=rgba_hwc_np_u8,
    )

    bigger_width = (width + 3) // 4 * 4
    bigger_height = (height + 3) // 4 * 4

    sanitized_rgb_hwc_np_u8 = np.ones(
        shape=(bigger_height, bigger_width, 3),
        dtype=np.uint8
    ) * 255

    sanitized_rgb_hwc_np_u8[:height, :width, :] = rgb_hwc_np_u8
    prii(sanitized_rgb_hwc_np_u8)

    # one way or another, get the image written to a file
    # as an rgb image whose shape is a multiple of 4 in both dimensions.
    
    image_abs_file_path = get_a_temp_file_path(suffix=".png")

    write_rgb_hwc_np_u8_to_png(
        rgb_hwc_np_u8=sanitized_rgb_hwc_np_u8,
        out_abs_file_path=image_abs_file_path,
    )

    print(f"image_abs_file_path: {image_abs_file_path}")
   
    json_file_path = get_a_temp_file_path(suffix=".json")

    subprocess.run(
        args=[
            str(executable_path),
            str(image_abs_file_path),
            instructions_string,
            str(json_file_path),
        ],
        check=True
    )

    jsonable = bj.load(json_file_path)
    xys = []
    for dct in jsonable:
        x = dct["x_pixel"]
        y = dct["y_pixel"]
        xys.append([x, y])
    return xys
