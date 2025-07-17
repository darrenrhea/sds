from prii import (
     prii
)
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from make_rgba_from_original_and_mask_paths import (
     make_rgba_from_original_and_mask_paths
)
from pathlib import Path
import pprint as pp
from infer_arbitrary_frames import (
     infer_arbitrary_frames
)
import os
import shutil
import numpy as np


final_model_id = "nbaplus2balplus1rwandaplus73southafricacaliboverlayepoch10" # os.environ["m"]
input_dir = Path("/shared/withscorebugs2")
output_dir = Path("/shared/temp")
output_dir.mkdir(exist_ok=True, parents=True)
input_paths = [
    x
    for x in input_dir.glob("*_original.jpg")
]

for input_path in input_paths:
    shutil.copy(src=input_path, dst=output_dir)

list_of_input_and_output_file_paths = [
    (
        x,
        output_dir / f"{x.stem[:-9]}_{final_model_id}.png"
    )
    for x in input_paths
]

np.random.shuffle(list_of_input_and_output_file_paths)
# list_of_input_and_output_file_paths = list_of_input_and_output_file_paths[:10]
pp.pprint(list_of_input_and_output_file_paths)

# this writes gray-scale segmentation masks:
infer_arbitrary_frames(
    final_model_id=final_model_id,
    list_of_input_and_output_file_paths=list_of_input_and_output_file_paths
)

for input_path, output_path in list_of_input_and_output_file_paths:
    rgba_hwc_np_u8 = make_rgba_from_original_and_mask_paths(
        original_path=input_path,
        mask_path=output_path,
        flip_mask=False,
        quantize=False
    )
    prii(rgba_hwc_np_u8)
    write_rgba_hwc_np_u8_to_png(
        rgba_hwc_np_u8=rgba_hwc_np_u8,
        out_abs_file_path=output_path,
        verbose=True
    )