import better_json as bj
import subprocess

from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)

from get_a_temp_file_path import (
     get_a_temp_file_path
)

from pathlib import Path
from typing import Optional, Tuple
import numpy as np


def get_click_on_image(
    image_path: Optional[Path] = None,
    rgb_hwc_np_u8: Optional[np.ndarray] = None,
    instructions_string: str = None
) -> Optional[Tuple[int, int]]:
    """
    Given an image, this throws it up on the screen for a user to click on a single point,
    or refuse by pressing space.

    - image_path: Path to the image.
    - hwc_rgb_np_u8: HWC RGB numpy array of uint8.

    Returns:
    Either the click as [x, y] in pixels, or None if the user presses Spacebar.
    """
    assert image_path is None or rgb_hwc_np_u8 is None, "image_path and rgb_hwc_np_u8 cannot both be given."
    assert image_path is not None or rgb_hwc_np_u8 is not None, "one of image_path and rgb_hwc_np_u8 must be given."

    executable_path = Path(
        "~/init/bin/one_click_on_image.exe"
    ).expanduser().resolve()

    assert executable_path.is_file(), f"executable_path not found. {executable_path} Did you install annotation_app?"

    if image_path is not None:
        assert image_path.is_file(), f"image_path not found. {image_path}"
    
    # one way or another, get the image written to a file
    # as an rgb image whose shape is a multiple of 4 in both dimensions.
    if image_path is None:
        image_path = get_a_temp_file_path(suffix=".png")

        write_rgb_hwc_np_u8_to_png(
            rgb_hwc_np_u8=rgb_hwc_np_u8,
            out_abs_file_path=image_path,
        )

    print(f"image_path: {image_path}")
   
    json_file_path = get_a_temp_file_path(suffix=".json")
    if instructions_string is None:
        instructions_string = "No instructions given."

    args = [
        str(executable_path),
        str(image_path),
        instructions_string,
        str(json_file_path),
    ]
  
    subprocess.run(
        args=args,
        check=True
    )

    jsonable = bj.load(json_file_path)
    assert isinstance(jsonable, list), f"jsonable is not a list. {jsonable}"
    if len(jsonable) == 0:
        return None
    
    xys = []
    for dct in jsonable:
        x = dct["x_pixel"]
        y = dct["y_pixel"]
        xys.append([x, y])
    xy = xys[0]
    return xy
