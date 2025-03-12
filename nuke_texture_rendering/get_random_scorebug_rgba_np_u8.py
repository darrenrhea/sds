from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)

from pathlib import Path
import numpy as np



def get_random_scorebug_rgba_np_u8(
    asset_repos_dir: Path,
):
    """
    TODO: take in multiple folders.
    """
    blank = np.random.rand() < 0.5
    if blank:
        return np.zeros((1080, 1920, 4), dtype=np.uint8)
    
    folder = asset_repos_dir / "bal_cutouts_approved/scorebugs"

    possible_file_paths = [
        x
        for x in folder.glob("*.png")
    ]
    
    index = np.random.randint(len(possible_file_paths))
    scorebug_image_file_path = possible_file_paths[index]

    scorebug_rgba = open_as_rgba_hwc_np_u8(scorebug_image_file_path)
    return scorebug_rgba
