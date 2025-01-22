from pathlib import Path

import numpy as np

import better_json as bj


def load_homography_and_rip_size_from_json(
        json_path: Path,
 ):
    """
    We had saved the homography want the way we flatten to infer to match the way we unflatten,
    so we save it for now.
    Move this to elsewhere later.
    """
    jsonable = bj.load(json_path)

    H = np.array(
        jsonable["homography_in_pixel_units"]
    )
    rip_height = int(jsonable["rip_height"])
    rip_width = int(jsonable["rip_width"])

    return dict(
        H=H,
        rip_height=rip_height,
        rip_width=rip_width
    )
