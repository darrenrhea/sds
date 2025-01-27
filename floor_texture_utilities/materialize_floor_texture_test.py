from get_a_random_floor_texture_jsonable_object_for_this_context import (
     get_a_random_floor_texture_jsonable_object_for_this_context
)
from materialize_floor_texture import (
     materialize_floor_texture
)
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from prii_linear_f32 import (
     prii_linear_f32
)


def test_materialize_random_floor_texture_1():
    """
    Getting a floor_texture is harder than you might think.
    It may need margin added.
    It needs to be positioned correctly in the world.
    Its colors need to be correct looking.
    It needs shadows and reflections added maybe.
    It may ultimately have to come from rendering a 3d model with lights.

    This gets a floor texture,
    which is both the rgb color information, and how it is placed into the world.
    """
    floor_id = "24-25_HOU_CORE"

    floor_texture_jsonable_object = get_a_random_floor_texture_jsonable_object_for_this_context(
        floor_id=floor_id,
    )
    
    ans = materialize_floor_texture(
        floor_texture_jsonable_object=floor_texture_jsonable_object,
        verbose=True,
    )

    color_corrected_texture_rgba_np_linear_f32 = ans["color_corrected_texture_rgba_np_linear_f32"]
    floor_placement_descriptor = ans["floor_placement_descriptor"]

    prii_linear_f32(color_corrected_texture_rgba_np_linear_f32)
    assert isinstance(floor_placement_descriptor, AdPlacementDescriptor)


if __name__ == "__main__":
    test_materialize_random_floor_texture_1()
