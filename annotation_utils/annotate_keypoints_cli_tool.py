from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_click_on_image_by_two_stage_zoom import (
     get_click_on_image_by_two_stage_zoom
)
import pprint
import sys
from attempt_to_annotate_a_fieldgoal import (
     attempt_to_annotate_a_fieldgoal
)
from attempt_to_annotate_a_cutout import (
     attempt_to_annotate_a_cutout
)
import argparse
from color_print_json import (
     color_print_json
)
from colorama import Fore, Style
import textwrap
from pathlib import Path
import better_json as bj


 
def annotate_keypoints_cli_tool():
    argp = argparse.ArgumentParser(
        description="annotate the scale of a bunch of cutouts of the same kind, team, league, and uniform colors",
        usage=textwrap.dedent(
            """\
            python annotate_keypoints ~/r/
            """
        )
    )
    argp.add_argument("dir_of_the_same_jersey", type=str)
    argp.add_argument("keypoint_name", type=str)
    opt = argp.parse_args()
    dir_of_one_kind = Path(opt.dir_of_the_same_jersey).resolve()
    keypoint_name = opt.keypoint_name
    assert dir_of_one_kind.is_dir(), f"{dir_of_one_kind} does not exist."

    original_paths = list(
        dir_of_one_kind.glob("*_original.jpg")
    )

    print(f"Going to annotate {len(original_paths)} images with the {keypoint_name=}, namely:")
    
    for original_path in original_paths:
        print(original_path)

    metainfos = []
    for original_path in original_paths:
        print(original_path.stem)
        L = len("_original")
        
        # get the part ending in six digits: 
        ends_on_sixdigits = original_path.stem[:-L]
        sixdigits = ends_on_sixdigits[-6:]
        clip_id = ends_on_sixdigits[: -7]
        assert len(sixdigits) == 6
        frame_index = int(sixdigits)
        metainfo = dict(
            clip_id=clip_id,
            frame_index=frame_index,
            original_path=original_path,
        )
        
        metainfos.append(metainfo)

    metainfos = sorted(metainfos, key=lambda x: (x["clip_id"], x["frame_index"]))

    for metainfo in metainfos:
        pprint.pprint(metainfo)

    for metainfo in metainfos:
        clip_id = metainfo["clip_id"]
        frame_index = metainfo["frame_index"]
        original_path = metainfo["original_path"]

        print(f"{original_path=}")
        out_path = original_path.with_suffix(".json")
        print(f"{Fore.CYAN}Does {out_path=} already exist?{Style.RESET_ALL}")
       

        if out_path.exists():
            result = bj.load(out_path)
        else:
            result = {}
        if "name_to_xy" in result:
            name_to_xy = result["name_to_xy"]
        else:
            name_to_xy = {}
        
        if keypoint_name in name_to_xy:
            print(
                textwrap.dedent(
                    f"""\
                    skipping {original_path} because {out_path} already specified {keypoint_name=}
                    If you really want a do-over, delete the json file. Or hand edit.
                    """
                )
            )
            continue

        rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(
            image_path=original_path,
        )

        keypoint_id_to_instruction = dict(
            tlu="Please click on the top left corner of the field goal",
            blu="Please click on the bottom left corner of the field goal",
            bru="Please click on the bottom right corner of the field goal",
            tru="Please click on the top right corner of the field goal",
        )

        instructions_string = keypoint_id_to_instruction[keypoint_name]

        click = get_click_on_image_by_two_stage_zoom(
            max_display_width=1920,
            max_display_height=1080,
            rgb_hwc_np_u8=rgb_hwc_np_u8,
            instructions_string=instructions_string,
        )

        print(click)
       
        name_to_xy[keypoint_name] = click
        result["clip_id"] = clip_id
        result["frame_index"] = frame_index
        result["name_to_xy"] = name_to_xy

        bj.color_print_json(result)
        print(f"writing {out_path}")
        bj.dump(obj=result, fp=out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    annotate_keypoints_cli_tool()
    print("Done.")