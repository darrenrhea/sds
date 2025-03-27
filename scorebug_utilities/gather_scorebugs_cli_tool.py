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



def gather_scorebugs_cli_tool():
    """
    Somebody or something needs to decide on which scorebugs
    we want the fake data to protect.
    This function is a CLI tool that helps do that.

    scorebug_config=$(gather_scorebugs bal_scorebugs)
    scorebug_config=$(gather_scorebugs hou_scorebugs)
    """
    argparser = argparse.ArgumentParser()
    valid_scorebug_contexts = ["bal_scorebugs", "hou_scorebugs"]
    argparser.add_argument(
        "context",
        help=f"The context in which this function is being called. Must be one of: {valid_scorebug_contexts}",
    )
    
    opt = argparser.parse_args()
    context = opt.context
    

    assert context in valid_scorebug_contexts, f"Invalid context: {context}, valid contexts: {valid_scorebug_contexts}"
    asset_repos_dir = Path("~/r").expanduser()
    
    if context == "bal_scorebugs":
        folders = [
            asset_repos_dir / "bal_cutouts_approved/scorebugs",
        ]
    elif context == "hou_scorebugs":
        folders = [
            asset_repos_dir / "houston_cutouts_approved/scorebugs",
        ]
    else:
        raise ValueError(f"Invalid context: {context}")
    
    for folder in folders:
        assert folder.exists(), f"{folder} does not exist."
        assert folder.is_dir(), f"{folder} is not a directory."
    
    file_paths = []
    for folder in folders:
        file_paths.extend(list(folder.glob("*.png")))

    if len(file_paths) == 0:
        print_red("No scorebug images found!")
        print("Adding a blank scorebug image.")
    
        blank_scorebug_file_path = get_a_temp_dir_path() / "the_trivial_scorebug.png"
        rgba_blank = np.zeros((1080, 1920, 4), dtype=np.uint8)
        
        write_rgba_hwc_np_u8_to_png(
            rgba_hwc_np_u8=rgba_blank,
            out_abs_file_path=blank_scorebug_file_path,
        )

        file_paths.append(blank_scorebug_file_path)
    
    for scorebug_image_file_path in file_paths:
        scorebug_rgba = open_as_rgba_hwc_np_u8(scorebug_image_file_path)
        prii(scorebug_rgba)
    
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



