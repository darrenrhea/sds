from get_cutout_descriptors_from_directories import (
     get_cutout_descriptors_from_directories
)
from pathlib import Path
import sys
from get_valid_cutout_kinds import (
     get_valid_cutout_kinds
)
from get_human_cutout_kinds import (
     get_human_cutout_kinds
)
import numpy as np
from open_as_hwc_rgba_np_uint8 import (
     open_as_hwc_rgba_np_uint8
)
from PasteableCutout import (
     PasteableCutout
)
from typing import (
     List
)
from collections import defaultdict

from prii import prii


def get_cutouts(
    sport: str,
    league: str,
    jersey_dir: Path,
    cutout_dirs: List[Path],
    diminish_for_debugging: bool = False,
    just_this_kind: str = None
) -> List[PasteableCutout]:
    """
    This does the actual cutting out of the cutouts.
    Returns a flattened list of cutouts,
    each with:
    a cutout_rgba_np_u8,
    a kind, such as
    referee
    or
    player
    or
    ball
    or
    led_screen_occluding_object
    file, the file the cutout is from for debugging
    bbox for debugging
    or complaining that someone cut out that player badly.
    """
    valid_cutout_kinds = get_valid_cutout_kinds(
        sport=sport,
    )
    cutouts = []
 
    all_cutout_descriptors = get_cutout_descriptors_from_directories(
        cutout_dirs=cutout_dirs,
        jersey_dir=jersey_dir,
    )
    kind_to_cutout_descriptors = dict()
    kind_to_num = dict()

    for kind in valid_cutout_kinds:
        kind_to_cutout_descriptors[kind] = [
            x
            for x in all_cutout_descriptors
            if x["metadata"]["kind"] == kind
        ]
        kind_to_num[kind] = len(kind_to_cutout_descriptors[kind])
        print(f"Found {kind_to_num[kind]} cutouts of kind {kind}.")


    if just_this_kind is not None:
        for kind in valid_cutout_kinds:
            if kind != just_this_kind:
                kind_to_num[kind] = 0
    
    if diminish_for_debugging:
        for kind in valid_cutout_kinds:
                kind_to_num[kind] = 2
   
    

    not_yet_filtered_cutout_descriptors = []
    for cutout_kind in valid_cutout_kinds:
        not_yet_filtered_cutout_descriptors.extend(
            kind_to_cutout_descriptors[cutout_kind][:kind_to_num[cutout_kind]]
        )

    # filter by uniform color:
    cutout_descriptors = []
    for desc in not_yet_filtered_cutout_descriptors:
        metadata = desc["metadata"]
        kind = metadata["kind"]
        assert kind in valid_cutout_kinds, f"{kind=} not in {valid_cutout_kinds=}"

        
        if kind == "player":
            major_uniform_color = metadata.get("uniform_major_color")
            assert major_uniform_color is not None, f"no major uniform color for {desc['png_path']}"

            if True or major_uniform_color in ["white", "maroon"]:
                cutout_descriptors.append(desc)
        else:
            cutout_descriptors.append(desc)
        
            
    human_cutout_kinds = get_human_cutout_kinds(league=league)

  
    kind_to_count = defaultdict(int)
    for cutout_descriptor in cutout_descriptors:
        # would be overkill to check these exist,
        # as get_cutout_descriptors() already checked them for existence:
        png_path = cutout_descriptor["png_path"]
        json_path = cutout_descriptor["json_path"]
        cutout_metadata = cutout_descriptor["metadata"]

      
        fullsize_rgba_np_u8 = open_as_hwc_rgba_np_uint8(
            image_path=png_path
        )
        # check for pngs that don't really have an alpha channel.
        percent_transparent = np.mean(
            (fullsize_rgba_np_u8[:, :, 3] < 128).astype(np.float32)
        )
        if percent_transparent < 0.01:
            print(f"Skipping {png_path} because it is less than 1% transparent.")
            sys.exit(1)
        
        name_to_xy = cutout_metadata["name_to_xy"]
        kind = cutout_metadata["kind"]
        if kind in ["pitcher"]:
            interfering_point = name_to_xy["interfering_point"]
            six_feet_below_that = name_to_xy["six_feet_below_that"]
            how_many_pixels_is_six_feet = six_feet_below_that[1] - interfering_point[1]
            # this is coming back quite large, but we think it still works if you go smaller:
        elif kind in ["batter"]:
            interfering_point = name_to_xy["interfering_point"]
            headtop = name_to_xy["headtop"]
            six_feet_below_that = name_to_xy["six_feet_below_that"]
            how_many_pixels_is_six_feet = six_feet_below_that[1] - headtop[1]
            # this is coming back quite large, but we think it still works if you go smaller:
        elif kind in human_cutout_kinds:
            bottom_of_lowest_foot = name_to_xy["bottom_of_lowest_foot"]
            six_feet_above_that = name_to_xy["six_feet_above_that"]
            how_many_pixels_is_six_feet = bottom_of_lowest_foot[1] - six_feet_above_that[1]
            # this is coming back quite large, but we think it still works if you go smaller:
        elif kind == "ball" or kind == "baseball":
            ball_top = name_to_xy["ball_top"]
            ball_bottom = name_to_xy["ball_bottom"]
            ball_center = name_to_xy["ball_center"]
            if kind == "ball":
                how_many_pixels_is_six_feet = (ball_bottom[1] - ball_top[1]) * 6
            elif kind == "baseball":
                how_many_pixels_is_six_feet = np.random.randint(1000, 1300)

        elif kind == "led_screen_occluding_object":
            top_of_led_screen = name_to_xy["top_of_led_screen"]
            bottom_of_led_screen = name_to_xy["bottom_of_led_screen"]
            # TODO: 2.75 is a guess:
            how_many_pixels_is_six_feet = (bottom_of_led_screen[1] - top_of_led_screen[1]) / 2.75 * 6

        
        rgba_np_u8 = fullsize_rgba_np_u8.copy()

        pasteable_cutout = PasteableCutout()
        pasteable_cutout.rgba_np_u8 = rgba_np_u8
        pasteable_cutout.kind = kind
        pasteable_cutout.file = png_path.name
        if kind in human_cutout_kinds and league in ["euroleague", "nba"]:
            pasteable_cutout.foot_bottom = bottom_of_lowest_foot
        elif kind in ["pitcher", "batter"] and league == "mlb":
            pasteable_cutout.interfering_point = interfering_point
        else:
            pasteable_cutout.ball_center = ball_center

        pasteable_cutout.how_many_pixels_is_six_feet = how_many_pixels_is_six_feet
        pasteable_cutout.metadata = cutout_metadata
        cutouts.append(pasteable_cutout)
        kind_to_count[kind] += 1

    if sport == "basketball":
        assert kind_to_count["referee"] >= 1, "no referees found?"
        assert kind_to_count["player"] >= 1, "no players found?"
    if sport == "baseball":
        assert kind_to_count["pitcher"] >= 1, "no pitchers found?"
    return cutouts


if __name__ == "__main__":
     
    asset_repos_dir = Path("~/r").expanduser()

    jersey_dir = asset_repos_dir / "jersey_ids"

    cutout_dirs_str = [
        "nba_misc_cutouts_approved/coaches",
        "nba_misc_cutouts_approved/coach_kidd",
        "nba_misc_cutouts_approved/randos",
        # "nba_misc_cutouts_approved/referees",  now that we are doing BAL we need yellow shirt referees
        "bal_cutouts_approved/referees",
        "nba_misc_cutouts_approved/balls",
        "nba_misc_cutouts_approved/objects",
        "allstar2025_cutouts_approved/phx_lightblue",
        "denver_nuggets_cutouts_approved/icon",
        "denver_nuggets_cutouts_approved/statement",
        "houston_cutouts_approved/icon",
    ]
    cutout_dirs = [
        asset_repos_dir / x
        for x in cutout_dirs_str
    ]

    cutouts = get_cutouts(
        sport="basketball",
        league="nba",
        jersey_dir=jersey_dir,
        cutout_dirs=cutout_dirs,
        diminish_for_debugging=False,
    )
    np.random.shuffle(cutouts)

    for cutout in cutouts:
        prii(cutout.rgba_np_u8)