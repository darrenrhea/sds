from pathlib import Path
from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)

import numpy as np

out_abs_file_path = Path(
    "~/r/munich_led_videos/solidblack/fake_movie/00000.png"
).expanduser()

out_abs_file_path.parent.mkdir(exist_ok=True, parents=True)

rgb = np.zeros((144, 1152, 3), dtype=np.uint8)
rgb[:, :, 0] = 52
rgb[:, :, 1] = 46
rgb[:, :, 2] = 48

rgb[:, :, 0] = 42
rgb[:, :, 1] = 36
rgb[:, :, 2] = 38

rgb[:, :, 0] = 32
rgb[:, :, 1] = 26
rgb[:, :, 2] = 28
write_rgb_hwc_np_u8_to_png(
    rgb_hwc_np_u8=rgb,
    out_abs_file_path=out_abs_file_path
)

print(f"pri {out_abs_file_path}")