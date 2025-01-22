from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from open_as_rgba_hwc_np_u8 import (
     open_as_rgba_hwc_np_u8
)
import better_json as bj
from pathlib import Path
import sys


def laacitd_look_at_all_cutouts_in_this_directory_cli_tool():

    if len(sys.argv) < 2:
        dir_path = Path.cwd()
    else:
        dir_path = Path(sys.argv[1])

    assert dir_path.is_dir(), f"{dir_path=} is not a directory"

    for p in dir_path.rglob("*.png"):
        image = open_as_rgba_hwc_np_u8(p)
        json_path = p.with_suffix(".json")
        if not json_path.exists():
            continue

        obj = bj.load(json_path)

        name_to_xy = obj["name_to_xy"]

        print(p)
        print(json_path)

        prii_named_xy_points_on_image(
            name_to_xy=name_to_xy,
            image=image,
            output_image_file_path=None,
            default_color=(0, 255, 0),  # green is the default
            dont_show=False,
        )