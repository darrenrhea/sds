import pprint as pp
from pathlib import Path
from prii import (
     prii
)
from get_cutouts import (
     get_cutouts
)
import numpy as np


def test_get_cutouts_1():
    """
    Returns a flattened list or cutouts,
    each with a cutout_rgba_np_u8, a kind
    like referee or player for filtering, file and bbox for debugging
    or complaining that someone cut out that player badly.
    """
    cutouts = get_cutouts(
        sport="basketball",
        league="nba",
        cutout_dirs=[
            Path("/shared/r/nba_misc_cutouts_approved/referees").expanduser(),
            Path("/shared/r/nba_misc_cutouts_approved/balls").expanduser(),
            Path("/shared/r/nba_misc_cutouts_approved/objects").expanduser(),
            Path("/shared/r/dallas_mavericks_cutouts_approved/association").expanduser(),
        ]
    )
    for cutout in cutouts:
        pp.pprint(cutout)
        # assert "kind" in cutout
        # assert "bbox" in cutout
        # assert "file" in cutout
        # assert "rgba_np_u8" in cutout
        kind = cutout.kind
        # bbox = cutout["bbox"]
        cutout_rgba_np_u8 = cutout.rgba_np_u8
        file = cutout.file
        metadata = cutout.metadata
        pp.pprint(metadata)
        assert "name_to_xy" in metadata
        name_to_xy = metadata["name_to_xy"]
        if kind in ["player", "referee", "coach", "randos"]:
            assert "bottom_of_lowest_foot" in name_to_xy
            assert "six_feet_above_that" in name_to_xy
        if kind == "ball":
            assert "ball_bottom" in name_to_xy
            assert "ball_top" in name_to_xy
            assert "ball_center" in name_to_xy
        if kind == "led_screen_occluding_object":
            assert "bottom_of_led_screen" in name_to_xy
            assert "top_of_led_screen" in name_to_xy
        assert isinstance(kind, str)
        # assert isinstance(cutout["bbox"], dict)
        assert isinstance(file, str)
        assert isinstance(cutout_rgba_np_u8, np.ndarray)
        assert cutout_rgba_np_u8.ndim == 3
        assert cutout_rgba_np_u8.shape[2] == 4
        assert cutout_rgba_np_u8.dtype == np.uint8


        print(f"{file=}")
        # print(f"{bbox=}")
        print(f"{kind=}")
        prii(cutout_rgba_np_u8)

if __name__ == "__main__":
    test_get_cutouts_1()
    print("get_cutouts_test.py has passed all tests")