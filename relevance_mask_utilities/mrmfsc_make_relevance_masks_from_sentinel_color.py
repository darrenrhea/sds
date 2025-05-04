from print_green import (
     print_green
)
from write_grayscale_hw_np_u8_to_png import (
     write_grayscale_hw_np_u8_to_png
)
from print_yellow import (
     print_yellow
)
from prii import (
     prii
)
import numpy as np
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from pathlib import Path


dataset_folder_strs = [
    "/shared/nfl-59773-skycam-ddv3_flat_wall/.approved",
]

dataset_folders = [
    Path(dataset_folder_str).expanduser().resolve()
    for dataset_folder_str in dataset_folder_strs
]

for dataset_folder in dataset_folders:
    assert dataset_folder.is_dir(), f"{dataset_folder} should be a directory"

for dataset_folder in dataset_folders:
    print_yellow(f"{dataset_folder=}")
    for p in dataset_folder.glob("*_original.jpg"):
        rgb = open_as_rgb_hwc_np_u8(
            image_path=p,
        )
        prii(rgb)
        relevance_mask = np.zeros(
            shape=rgb.shape[:2],
            dtype=np.uint8
        )
        relevance_mask[:, :] = 255 * np.logical_not((rgb == 0).all(axis=2))
        prii(relevance_mask)
        out_abs_file_path = p.with_suffix(
            ".relevance.png"
        )
        len_to_remove = len("_original.jpg")
        out_name = p.name[:-len_to_remove] + "_relevance.png"
        out_abs_file_path = p.parent / out_name
        print_yellow(f"{out_abs_file_path}")
        write_grayscale_hw_np_u8_to_png(
            grayscale_hw_np_u8=relevance_mask,
            out_abs_file_path=out_abs_file_path,
        )
  
print_green("for x in /shared/nfl-59773-skycam-ddv3_flat_wall/.approved/* ; do pri $x ; done")