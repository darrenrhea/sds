from pathlib import Path
from get_all_floor_texture_jsonable_objects import (
     get_all_floor_texture_jsonable_objects
)
import numpy as np


def get_a_random_floor_texture_jsonable_object_for_this_context(
    floor_id: str,
    asset_repos_dir: Path,
) -> dict:
    """
    This simply gets one of the hopefully many floor texture jsonable objects that is available for this floor_id.

    Jsonable records that describe each floor texture should be stored somewhere,
    like a database or the file system,
    so that they can be retrieved by floor_id.

    Getting a floor_texture is harder than you might think.
    It may need constant color margin added to be big enough to cover the whole floor.
    It needs to be positioned correctly in the world.
    Its colors need to be correct looking.
    It needs shadows and reflections added maybe.
    It may ultimately have to come from rendering a 3d model with lights.

    This gets a floor texture,
    which is uncorrected rgb color information,
    and how it is placed into the world according to the where the interior top_left_interior_court_corner_xy and bottom_right_interior_court_corner_xy are.

    """
    assert isinstance(floor_id, str)
    floor_texture_jsonable_objects = get_all_floor_texture_jsonable_objects(
        asset_repos_dir=asset_repos_dir,
    )
    
    possibilities = [
        x for x in floor_texture_jsonable_objects
        if (
            x["floor_id"] in [floor_id]
            and
            x.get("approved", False)
        )
    ]
    assert len(possibilities) > 0, "ERROR: no approved possibilities found for this {floor_id=}"

    r = np.random.randint(0, len(possibilities))
    possibility = possibilities[r]
    
    return possibility

