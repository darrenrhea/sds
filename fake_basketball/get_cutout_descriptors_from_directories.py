from typing import List
from color_print_json import (
     color_print_json
)
from assert_cutout_metadata_is_good import (
     assert_cutout_metadata_is_good
)

from pathlib import Path
from collections import defaultdict
import better_json as bj


def get_cutout_descriptors_from_directories(
    jersey_dir: Path,
    cutout_dirs: List[Path]
):
    """
    We give up on more complicated things and simply say
    it is whatever is in these directories.
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

   
    for p in cutout_dirs:
        assert p.is_dir(), f"{p} is not a directory"
        assert p.is_absolute(), f"{p} is not absolute"


   

    for dir_of_one_kind in cutout_dirs:
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
                jersey_json_path =  jersey_dir / f"{jersey_id}.json5"
                jersey_info = bj.load(jersey_json_path)
                color_print_json(jersey_info)
                kind = jersey_info["kind"]
                uniform_major_color = jersey_info["major_color"]
                uniform_minor_color = jersey_info["minor_color"]
                team = jersey_info.get("team_id")
                if kind == "player":
                    assert team is not None, f"no team for {jersey_id=}"
                cutout_metadata["league"] = jersey_info["league_id"]
                cutout_metadata["kind"] = kind
                cutout_metadata["team"] = team
                cutout_metadata["uniform_major_color"] = uniform_major_color
                cutout_metadata["uniform_minor_color"] = uniform_minor_color


            
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
