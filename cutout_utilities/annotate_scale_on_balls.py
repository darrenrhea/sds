from colorama import Fore, Style
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from get_clicks_on_image import ( 
     get_clicks_on_image
)

from pathlib import Path
import better_json as bj


def attempt_to_annotate_a_cutout(
    cutout_path: Path,
):
    while True:
        print(
            f"{Fore.YELLOW}click bottom of ball, then center of ball, then top of ball. Then press spacebar to finish.{Style.RESET_ALL}"
        )
        points = get_clicks_on_image(
            image_path=cutout_path,
            rgba_hwc_np_u8=None,
            instructions_string="press spacebar to finish."
        )

        if len(points) != 3:
            print(f"You were supposed to click 3 points, but you clicked {len(points)} times!. Try again.")
            continue
        
        # If we get here, we have 3 points.
        name_to_xy = {
            "ball_bottom": points[0],
            "ball_center": points[1],
            "ball_top": points[2],
        }
        
        prii_named_xy_points_on_image(
            image=cutout_path,
            name_to_xy=name_to_xy
        )

        ans = input("type y if you happy with this, or s to skip this one, or n to redo: (N/s/y)")

        if ans.lower() == "y":
            break

        if ans.lower() == "s":
            return None
        
        print("Try again please:")
    
    return name_to_xy
    
   

 
if __name__ == "__main__":
    kind = "baseball"
    subfolder = "balls"
    league="mlb"
    ball_kind = "mlb"

    dir_of_one_kind = Path(
        "~/r/brewcub_cutouts_approved/baseballs"
    ).expanduser()

    cutout_paths = list(
        dir_of_one_kind.glob("*.png")
    )

    print(f"Going to annotate {len(cutout_paths)} cutouts of {kind=}, namely:")
    
    for cutout_path in cutout_paths:
        print(cutout_path)

    for cutout_path in cutout_paths:
        print(f"{cutout_path=}")
        out_path = cutout_path.with_suffix(".json")
        print(f"{out_path=}")
        if out_path.exists():
            print(f"skipping {cutout_path} because {out_path} already exists")
            continue

        metainfo = dict(
            kind=kind,
            league=league,
            ball_kind=ball_kind,
            name_to_xy={}
        )


        name_to_xy = attempt_to_annotate_a_cutout(cutout_path)

        if name_to_xy is not None:
            metainfo["name_to_xy"] = name_to_xy
            bj.color_print_json(metainfo)
            bj.dump(obj=metainfo, fp=out_path)
            print(f"wrote {out_path}")
