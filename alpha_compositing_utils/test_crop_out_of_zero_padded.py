import PIL.Image
import numpy as np
from image_openers import open_as_hwc_rgba_np_uint8
from pathlib import Path
from print_image_in_iterm2 import print_image_in_iterm2

from crop_out_of_zero_padded import crop_out_of_zero_padded


def test_crop_out_of_zero_padded():
    image_path = Path(
        f"~/r/synthetic_baseball/baseball_cutouts/baseball.png"
    ).expanduser()
    
    x = open_as_hwc_rgba_np_uint8(
        image_path=image_path
    )
    
    original_height = x.shape[0]
    original_width = x.shape[1]

    print(f"consider this {original_height=} x {original_width=} image:")
    print_image_in_iterm2(
        rgba_np_uint8=
        x
    )

    crop_height = np.random.randint(
        low=0,
        high=original_height * 4
    )

    crop_width = np.random.randint(
        low=0,
        high=original_width * 4
    )

    i_min = np.random.randint(
        low=-2 * original_height,
        high=2 * original_height
    )

    j_min = np.random.randint(
        low=-2 * original_width,
        high=2 * original_width
    )

    i_max = i_min + crop_height
    j_max = j_min + crop_width


    crop = crop_out_of_zero_padded(
        x=x,
        i_min=i_min,
        i_max=i_max,
        j_min=j_min,
        j_max=j_max,
    )

    print(f"We cropped out of a {original_height} x {original_width} image the range [{i_min=}, {i_max=}) x [{j_min=}, {j_max=})")
    
    print_image_in_iterm2(
        rgba_np_uint8=crop
    )

    assert crop.shape[0] == crop_height
    assert crop.shape[1] == crop_width
    assert crop.dtype == x.dtype
    assert crop.shape[2:] == x.shape[2:]

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):

            if (
                i >=0 and i < original_height
                and j >=0 and j < original_width
            ):
                assert np.all(crop[i - i_min, j - j_min, ...] == x[i, j, ...])
            else:
                assert np.all(crop[i - i_min, j - j_min, ...] == 0), f"{crop[i - i_min, j - j_min, ...]}"

  
if __name__ == "__main__":
    test_crop_out_of_zero_padded()
