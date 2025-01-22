from color_print_json import (
     color_print_json
)
from assert_cutout_metadata_is_good import (
     assert_cutout_metadata_is_good
)

from pathlib import Path
from collections import defaultdict
import better_json as bj


def get_cutout_descriptors(
    context_id: str
):
    """
    This gathers cutouts appropriate for the context
    for a bunch of folders.
    In modern times,
    a cutout is an RGBA png image cutting out the whatever:
    player, referee, coach, ball, other things like mascots.
    Each cutout has an associated json file with it that has:
    a kind (must be in player, referee, coach, ball, mascot, etc.)
    and possibly a toe_point
    and possibly a head_point
    and a scale (what is 6 feet tall in its pixels?)    
    """
    kind_to_count = defaultdict(int)
    
    cutout_descriptors = []

    if context_id == "london":
        subdirs = [
            "north_players",
            "south_players",
            "balls",
            "referees",
            "coaches",
        ]
        abs_dirs = [
            Path(f"~/r/london_cutouts_approved/{subdir}").expanduser()
            for subdir in subdirs
        ]
    elif context_id == "munich":
        subdirs = [
            "referees_faithful",
            "coaches_faithful",
            "white_uniforms",
            "maroon_uniforms",
            "balls",
        ]
        abs_dirs = [
            Path(f"~/r/munich_cutouts_approved/{subdir}").expanduser()
            for subdir in subdirs
        ]
    elif context_id == "maccabi":
        subdirs = [
            "referees_faithful",
            "coaches_faithful",
            "white_uniforms",
            "maroon_uniforms",
            "balls",
        ]
        abs_dirs = [
            Path(f"~/r/munich_cutouts_approved/{subdir}").expanduser()
            for subdir in subdirs
        ]

        abs_dirs.append(
            Path("~/r/maccabi_cutouts_approved").expanduser() / "yellow_with_blue"
        )
        abs_dirs.append(
            Path("~/r/maccabi_cutouts_approved").expanduser() / "blue_with_yellow"
        )
        abs_dirs.append(
            Path("~/r/maccabi_cutouts_approved").expanduser() / "black_with_blue_yellow"
        )
    elif context_id == "barcelona":
        subdirs = [
            "referees_faithful",
            "coaches_faithful",
            "white_uniforms",
            "maroon_uniforms",
            "balls",
        ]
        abs_dirs = [
            Path(f"~/r/munich_cutouts_approved/{subdir}").expanduser()
            for subdir in subdirs
        ]

        abs_dirs.append(
            Path("~/r/barcelona_cutouts_approved").expanduser() / "yellow_with_red_stripes"
        )
    elif context_id == "athens":
        subdirs = [
            "referees_faithful",
            "coaches_faithful",
            "maroon_uniform_shoes",
            # "white_uniforms",
            # "maroon_uniforms",
            "balls",
        ]
        abs_dirs = [
            Path(f"~/r/munich_cutouts_approved/{subdir}").expanduser()
            for subdir in subdirs
        ]
        # abs_dirs.append(
        #     Path("~/r/zalgiris_cutouts_approved").expanduser() / "white_with_green"
        # )
        # abs_dirs.append(
        #     Path("~/r/barcelona_cutouts_approved").expanduser() / "yellow_with_red_stripes"
        # )
        # abs_dirs.append(
        #     Path("~/r/boston_cutouts_approved").expanduser() / "white_with_green"
        # )



    for abs_dir in abs_dirs:
        assert abs_dir.is_dir(), f"{abs_dir} is not a directory"

    for dir_of_one_kind in abs_dirs:
        cutout_png_paths = list(
            dir_of_one_kind.glob("*.png")
        )

        for cutout_png_path in cutout_png_paths:
            print(cutout_png_path)
            cutout_metadata_path = cutout_png_path.with_suffix(".json")
            if not cutout_metadata_path.exists():
                print(f"skipping {cutout_png_path} because the associated {cutout_metadata_path} does not exist")
                continue
            cutout_metadata = bj.load(cutout_metadata_path)
           
            if "jersey_id" in cutout_metadata:
                jersey_id = cutout_metadata["jersey_id"]
                print(f"Jersey id: {jersey_id}")
                jersey_json_path = Path("~/r/jersey_ids/").expanduser() / f"{jersey_id}.json5"
                jersey_info = bj.load(jersey_json_path)
                color_print_json(jersey_info)
                cutout_metadata["uniform_major_color"] = jersey_info["major_color"]
                cutout_metadata["uniform_minor_color"] = jersey_info["minor_color"]
                cutout_metadata["team"] = jersey_info["team_id"]
                cutout_metadata["league"] = jersey_info["league_id"]
                cutout_metadata["kind"] = jersey_info["kind"]
            
            metadata = assert_cutout_metadata_is_good(cutout_metadata)
            
            cutout_descriptor = dict(
                metadata=metadata,
                png_path=cutout_png_path,
                json_path=cutout_metadata_path,
            )
            cutout_descriptors.append(cutout_descriptor)
            kind = metadata["kind"]
            kind_to_count[kind] += 1

    # assert kind_to_count["ball"] >= 1, "no balls found"
    return cutout_descriptors
