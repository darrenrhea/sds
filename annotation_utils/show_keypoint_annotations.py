from print_green import (
     print_green
)
from color_print_json import (
     color_print_json
)
from prii_named_xy_points_on_image_with_auto_zoom import (
     prii_named_xy_points_on_image_with_auto_zoom
)
from print_yellow import (
     print_yellow
)
from print_red import (
     print_red
)
import pprint
import sys
import argparse
import textwrap
from pathlib import Path
import better_json as bj


 
def show_keypoint_annotations():
    argp = argparse.ArgumentParser(
        description="annotate the scale of a bunch of cutouts of the same kind, team, league, and uniform colors",
        usage=textwrap.dedent(
            """\
            python annotate_keypoints ~/r/
            """
        )
    )
    argp.add_argument("folder_containing_jsons", type=str)
    # argp.add_argument("keypoint_name", type=str)
    opt = argp.parse_args()
    folder_containing_jsons = Path(opt.folder_containing_jsons).resolve()
    assert folder_containing_jsons.is_dir(), f"{folder_containing_jsons} does not exist."

    json_paths = list(
        folder_containing_jsons.glob("*_original.json")
    )

    counter = 0
    for json_path in json_paths:
        try:
            obj = bj.load(json_path)
        except:
            print_red(f"Error loading {json_path}")
            sys.exit(1)
        if "name_to_xy" not in obj:
            print_yellow(f"Skipping {json_path} because it does not have name_to_xy")
            continue

        clip_id = obj["clip_id"]
        frame_index = obj["frame_index"]
        name_to_xy = obj["name_to_xy"]
        keypoint_names = ["tlu", "blu", "bru", "tru"]
        has_all_keypoints = all(
            name_to_xy.get(keypoint_name) is not None
            for keypoint_name in keypoint_names
        )
        name_to_xy_extant = {
            keypoint_name: name_to_xy[keypoint_name]
            for keypoint_name in name_to_xy.keys()
            if name_to_xy[keypoint_name] is not None
        }
        if len(name_to_xy_extant) > 0:
            original_path = json_path.parent / f"{clip_id}_{frame_index:06d}_original.jpg"
            print(original_path)
            prii_named_xy_points_on_image_with_auto_zoom(
                name_to_xy=name_to_xy_extant,
                image=original_path,
                output_image_file_path=None,
                default_color=(255, 0, 255),
                dont_show=False,
            )
            color_print_json(name_to_xy)
        if has_all_keypoints:
            counter += 1

    print_green(f"Found {counter} keypoint annotations with all keypoints: {keypoint_names}")
if __name__ == "__main__":
    show_keypoint_annotations()
    print("Done.")