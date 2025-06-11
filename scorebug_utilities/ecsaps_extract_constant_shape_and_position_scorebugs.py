from write_rgb_hwc_np_u8_to_png import (
     write_rgb_hwc_np_u8_to_png
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import argparse
from color_print_json import color_print_json
import get_a_temp_dir_path
from print_green import print_green
from print_red import print_red
from sjaios_save_jsonable_as_its_own_sha256 import sjaios_save_jsonable_as_its_own_sha256
from write_rgba_hwc_np_u8_to_png import (
     write_rgba_hwc_np_u8_to_png
)
from prii import prii
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)

from pathlib import Path
import numpy as np
from store_file_by_sha256 import store_file_by_sha256



def ecsaps_extract_constant_shape_and_position_scorebugs():
    xmin = 464 # good
    xmax = 1460 # good
    ymin = 950  # good
    ymax = 1058  # good
    folders = [
        Path("~/a/preannotations/fixups/okc-ind-2025-06-05-youtube/pacersrev0epoch35").expanduser(),
    ]

    for folder in folders:
        assert folder.exists(), f"{folder} does not exist."
        assert folder.is_dir(), f"{folder} is not a directory."
    
    file_paths = []
    for folder in folders:
        file_paths.extend(list(folder.glob("*.jpg")))

  
    out_dir = Path("~/scorebugs").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for scorebug_image_file_path in file_paths:
        out_abs_file_path = out_dir / f"{scorebug_image_file_path.stem}_scorebug.png"
        rgb = open_as_rgb_hwc_np_u8(scorebug_image_file_path)
        rgb = rgb[ymin:ymax, xmin:xmax, :]
        prii(rgb)
        write_rgb_hwc_np_u8_to_png(
            rgb_hwc_np_u8=rgb,
            out_abs_file_path=out_abs_file_path,
        )
    

    dht_save = False
    if dht_save:
        sha256s = []
        for scorebug_image_file_path in file_paths:
            scorebug_sha256 = store_file_by_sha256(scorebug_image_file_path)
            sha256s.append(scorebug_sha256)
        
        for sha256 in sha256s:
            print_green(f"pri {sha256}")

        scorebugs = []
        for scorebug_image_file_path, scorebug_sha256 in zip(file_paths, sha256s):
            scorebug = {
                "scorebug_provenance": scorebug_image_file_path.name,
                "scorebug_sha256": scorebug_sha256,
            }
            print_green(scorebug_image_file_path)
            scorebugs.append(scorebug)

        jsonable = {
            "scorebugs": scorebugs,
        }
        
        color_print_json(jsonable)

        scorebug_config_sha256 = sjaios_save_jsonable_as_its_own_sha256(
            obj=jsonable,
        )

        print_green("This should work on any machine:")
        print_green(f"pri {scorebug_config_sha256}")
        
        print(scorebug_config_sha256)

if __name__ == "__main__":
   

    ecsaps_extract_constant_shape_and_position_scorebugs()
