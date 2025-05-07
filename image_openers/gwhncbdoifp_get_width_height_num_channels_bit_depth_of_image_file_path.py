from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def gwhncbdoifp_get_width_height_num_channels_bit_depth_of_image_file_path(
    image_file_path: str | Path
) -> Tuple[int, int, int, int]:
    """
    Return basic pixel-level metadata for *image_file_path*.

    Parameters
    ----------
    image_file_path
        Path to any image format that Pillow can open (JPEG, PNG, TIFF, GIF, …).

    Returns
    -------
    (width, height, n_channels, bit_depth)

      * **width**  – pixel width  
      * **height** – pixel height  
      * **n_channels** – 1 for gray / paletted, 3 for RGB, 4 for RGBA, etc.  
      * **bit_depth** – bits **per channel** (e.g. 8 for most JPEG/PNG, 16 for
        “I;16” PNGs, 1 for bilevel “1” mode)

    Notes
    -----
    * Conversion to a NumPy array is the most portable way to discover both the
      number of channels and the sample precision.  
    * For *paletted* images (“P” mode) Pillow expands the palette when the array
      is created, so you still receive the underlying gray/RGB data.
    """
    image_file_path = Path(image_file_path).expanduser().resolve()
    if not image_file_path.is_file():
        raise FileNotFoundError(image_file_path)

    with Image.open(image_file_path) as im:
        width, height = im.size

        # Convert to a NumPy array **without** altering the original mode
        arr = np.asarray(im)

        # 2-D → single-channel; 3-D → last axis is channel count
        n_channels = 1 if arr.ndim == 2 else arr.shape[2]

        # dtype.itemsize gives bytes → multiply by 8 for bits per channel
        bit_depth = arr.dtype.itemsize * 8

    return width, height, n_channels, bit_depth


# ---------------------------------------------------------------------------
# Quick CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        w, h, c, b = get_image_meta(sys.argv[1])
    except IndexError:
        sys.exit("usage: python get_image_meta.py <image_file>")
    print(f"{sys.argv[1]}: {w} × {h}  |  {c} channel(s)  |  {b}-bit")