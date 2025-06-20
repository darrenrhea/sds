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
    clip_id = "ind-okc-2025-06-11-hack"
    clip_id_mapsto_frame_indices = {
        "okc-ind-2025-06-05-youtube": [
            129,
            335,
            513,
            614,
            1152,
            1153,
            1154,
            1155,
            1156,
            1157,
            1158,
            1159,
            1160,
            1161,
            1162,
            1415,
            1416,
            1417,
            1418,
            1419,
            1420,
            1421,
            1422,
            1423,
            1424,
            1425,
            1426,
            1427,
            1428,
            1431,
            2075,
            2592,
            2604,
            2632,
            2806,
            2807,
            2808,
            2809,
            2810,
            2811,
            2812,
            2813,
            2814,
            2815,
            2816,
            2817,
            2818,
            2819,
            2820,
            2821,
            2822,
            2823,
            2824,
            2825,
            2826,
            2827,
            2828,
            #Darren:
            2100,
            2101,
            2102,
            2111,
            2112,
            2113,
            2114,
            2115,
            2116,
            2117,
            2214,
            2178,
            2179,
            2180,
            2181,
            2182,
            2183,
            2184,
            2185,
            2186,
            2187,
            2188,
            2189,
            2190,
            2191,
            2192,
            2193,
            2194,

            2267,
            2268,
            2269,
            2270,
            2271,
            2272,
            2273,
            2274,
            2275,
            2276,
            2277,
            2278,
            2279,
            2280,
            2281,
            2282,
            2283,
            2284,
            2285,
            2286,
            2287,
            2288,
            2289,
            2290,
            2291,
            2292,
            2293,
            2294,
            2607,
            2608,
            2609,
            2610,
            2611,
            2612,
            2613,
            2614,
            2615,
            2616,
            2617,
            2618,
            2619,
        ],
        "ind-okc-2025-06-11-hack": (
            list(range(3536, 3574+1))
            +
            list(range(3710, 3748+1))
            +
            list(range(4033, 4079+1))
            +
            [4371,4383,4506,4807,]
            +
        )
    }
    
    src_folder = Path(f"/mnt/nas/volume1/videos/clips/{clip_id}/frames")
    
    assert src_folder.exists(), f"{src_folder} does not exist."
    assert src_folder.is_dir(), f"{src_folder} is not a directory."
    
    file_paths = []
    for frame_index in frame_indices:
        file_path = src_folder / f"{clip_id}_{frame_index:06d}_original.jpg"
        assert file_path.exists(), f"{file_path} does not exist."
        assert file_path.is_file(), f"{file_path} is not a file."
        file_paths.append(file_path)
    
  
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
