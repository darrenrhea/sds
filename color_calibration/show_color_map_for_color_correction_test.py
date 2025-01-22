from color_print_json import (
     color_print_json
)
from show_color_map_for_color_correction import (
     show_color_map_for_color_correction
)
import numpy as np
from pathlib import Path
import better_json as bj
import pprint as pp


def test_show_color_map_for_color_correction_1():
    json5_path = Path("~/r/color_calibration/color_correction.json5").expanduser()
    color_pairs = bj.load(json5_path)
    num_matches = len(color_pairs)
    color_map = np.zeros((num_matches, 2, 3), dtype=np.uint8)
    show_color_map_for_color_correction(
         color_map
    )
    for i, dct in enumerate(color_pairs):
        color_print_json(dct)
        from_ = dct["led_png"]
        to_ = dct["footage_png"]
        color_map[i, 0, :] = from_
        color_map[i, 1, :] = to_

   
    show_color_map_for_color_correction(
       color_map
    )

if __name__ == "__main__":
    test_show_color_map_for_color_correction_1()