from pathlib import Path
from get_color_corrected_floor_texture_with_margin_added import (
     get_color_corrected_floor_texture_with_margin_added
)
from is_sha256 import (
     is_sha256
)
from typing import Any, Dict, List
from functools import lru_cache 
import better_json as bj

@lru_cache(maxsize=1, typed=False)
def get_all_floor_texture_jsonable_objects(
    slow_check: bool = False,
    only_approved: bool = True
) -> List[Dict[str, Any]]:
    """
    A list/set of jsonable records that describe each floor texture should be stored somewhere,
    like a database or the file system,
    so that they can be retrieved by floor_id.

    This gets them from there.  As they are inherently small, you might as well get all of them at once.

    Getting a floor_texture ready to be stuck underneath annotations is harder than you might think.
    It may need constant color margin added to be big enough to cover the whole floor.
    It needs to be positioned correctly in the world.
    Its colors need to be corrected.
    It should have shadows and reflections added.
    It may ultimately have to come from rendering a 3d model with lights.

    This gets a floor texture,
    which is uncorrected rgb color information,
    and how it is placed into the world according to the where the interior top_left_interior_court_corner_xy and bottom_right_interior_court_corner_xy are.

    """
    # this needs to be stored in a database, a repo, or the file system somewhere:
    database_file_path = Path("/shared/r/floor_texture_data/floor_texture_data.json5").expanduser()
    if not database_file_path.exists():
        raise FileNotFoundError(f"The database file {database_file_path} does not exist.")
    
    jsonable_objects = bj.load(database_file_path)
    
    if only_approved:
        jsonable_objects = [
            x
            for x in jsonable_objects
            if x.get("approved", False)
        ]

    for obj in jsonable_objects:
        texture_sha256 = obj["texture_sha256"]
        assert is_sha256(texture_sha256)
        uncorrected_margin_color_rgb = obj["uncorrected_margin_color_rgb"]
        
        assert len(uncorrected_margin_color_rgb) == 3
        for x in uncorrected_margin_color_rgb:
            assert isinstance(x, int)
            assert 0 <= x <= 255
        color_correction_sha256 = obj["color_correction_sha256"]
        assert is_sha256(color_correction_sha256)
        
        texture_width_in_pixels = obj["texture_width_in_pixels"]
        assert isinstance(texture_width_in_pixels, int)
        assert texture_width_in_pixels > 0
        texture_height_in_pixels = obj["texture_height_in_pixels"]
        assert isinstance(texture_height_in_pixels, int)
        assert texture_height_in_pixels > 0
        points = obj["points"]
        assert isinstance(points, dict)
        assert "top_left_interior_court_corner_xy" in points
        assert "bottom_right_interior_court_corner_xy" in points
        for key in points:
            assert isinstance(points[key], list)
            assert len(points[key]) == 2
            assert isinstance(points[key][0], int) or isinstance(points[key][0], float)
            assert isinstance(points[key][1], int) or isinstance(points[key][1], float)

        if slow_check:
            get_color_corrected_floor_texture_with_margin_added(
                floor_texture_jsonable_object=obj,
                verbose=True
            )

    return jsonable_objects
   
    
