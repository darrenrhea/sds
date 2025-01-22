from create_ij_displacement_and_weight_pairs import (
     create_ij_displacement_and_weight_pairs
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
from blur_both_original_and_mask_u8 import (
     blur_both_original_and_mask_u8
)
from pathlib import Path
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)

import numpy as np


def test_blur_both_original_and_mask_u8_1():
    mask_path = Path(
        "~/r/bos-mia-2024-04-21-mxf_led/justan/bos-mia-2024-04-21-mxf_495000_nonfloor.png"
    ).expanduser()
    assert mask_path.is_file()

    original_path = Path(
        "~/r/bos-mia-2024-04-21-mxf_led/justan/bos-mia-2024-04-21-mxf_495000_original.jpg"
    ).expanduser()
    assert original_path.is_file()

    rgba_np_u8 = make_rgba_from_original_and_mask_paths(
        original_path=original_path,
        mask_path=mask_path,
        flip_mask=False,
        quantize=False,
    )

    # rgba_np_u8 = np.array(
    #     [255, 255, 255, 255]
    # ).reshape(
    #      (1, 1, 4),
    # )

    ij_displacement_and_weight_pairs = create_ij_displacement_and_weight_pairs(
        radius=5
    )

    blurred_rgba_hwc_np_u8 = blur_both_original_and_mask_u8(
        rgba_np_u8=rgba_np_u8,
        ij_displacement_and_weight_pairs=ij_displacement_and_weight_pairs,
    )
    
    out_abs_file_path = Path(
        "temp.png"
    ).resolve()

    write_rgba_hwc_np_u8_to_png(
        rgba_hwc_np_u8=blurred_rgba_hwc_np_u8,
        out_abs_file_path=out_abs_file_path,
    )
    print(f"Wrote: {out_abs_file_path}")


if __name__ == "__main__":
    test_blur_both_original_and_mask_u8_1()