from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from pathlib import Path
from get_datapoint_path_tuples_from_list_of_dataset_folders import (
     get_datapoint_path_tuples_from_list_of_dataset_folders
)
import numpy as np
folder = Path(
    "~/r/nfl-59773-skycam-ddv3_flat_wall/.approved"
).expanduser()

datapoint_path_tuples = get_datapoint_path_tuples_from_list_of_dataset_folders(
    dataset_folders=[
        folder,
    ],
)
datapoint_path_tuple = datapoint_path_tuples[0]
original_path, mask_path, maybe_relevance_path = datapoint_path_tuple

rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(
    image_path=original_path
)
prii(rgb_hwc_np_u8)

rgb_hwc_np_f32 = rgb_hwc_np_u8.astype(np.float32)

radius = 3
centroid = np.array(
    [
        51, 58, 53
    ],
    dtype=np.float32
)

diff = rgb_hwc_np_f32 - centroid
squared = diff ** 2
squared_sum = np.sum(squared, axis=-1)
indicator = squared_sum < radius ** 2 

rgb_hwc_np_u8[indicator, :] = [255, 255, 0]
prii(rgb_hwc_np_u8)