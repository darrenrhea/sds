from make_placement_descriptor_for_nba_floor_texture import (
     make_placement_descriptor_for_nba_floor_texture
)
from get_color_corrected_floor_texture_with_margin_added import (
     get_color_corrected_floor_texture_with_margin_added
)
from get_a_random_floor_texture_jsonable_object_for_this_context import (
     get_a_random_floor_texture_jsonable_object_for_this_context
)
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)


def get_a_random_floor_texture_for_this_context(
    floor_id: str,
):
    """
    This is one of the most convenient ways to get a floor texture for a given floor_id.

    Is does most everything for you, except for adding shadows and lights augmentation.

    This gets a large numpy in RAM, margin-added, color-corrected floor texture with placement in the world, for this floor_id.

    Getting a floor_texture is harder than you might think.
    It may need margin added.
    It needs to be positioned correctly in the world.
    Its colors need to be correct-looking.
    It needs shadows and reflections added maybe.
    It may ultimately have to come from rendering a 3d model with lights.

    This gets a floor texture,
    which is both the rgb color information, and how it is placed into the world.
    """
    assert isinstance(floor_id, str)
    assert floor_id in [
        "24-25_HOU_CORE",
        "24-25_HOU_STMT",
        "24-25_HOU_CITY",
        "22-23_ATL_CORE",
        "22-23_CHI_CORE",
        "22-23_WAS_CORE",
    ]

   
    floor_texture_jsonable_object = get_a_random_floor_texture_jsonable_object_for_this_context(
        floor_id=floor_id
    )

    color_corrected_with_margin = get_color_corrected_floor_texture_with_margin_added(
        floor_texture_jsonable_object=floor_texture_jsonable_object,
    )
    color_corrected_texture_rgba_np_linear_f32 = color_corrected_with_margin["color_corrected_texture_rgba_np_linear_f32"]
    points = color_corrected_with_margin["points"]
    texture_height_in_pixels = color_corrected_texture_rgba_np_linear_f32.shape[0]
    texture_width_in_pixels = color_corrected_texture_rgba_np_linear_f32.shape[1]
    del color_corrected_with_margin

    floor_placement_descriptor = make_placement_descriptor_for_nba_floor_texture(
        points=points,
        texture_width_in_pixels=texture_width_in_pixels,
        texture_height_in_pixels=texture_height_in_pixels,
    )

    assert color_corrected_texture_rgba_np_linear_f32.shape[2] == 4
    assert color_corrected_texture_rgba_np_linear_f32.dtype == "float32"
    assert isinstance(floor_placement_descriptor, AdPlacementDescriptor)

    answer = dict(
        color_corrected_texture_rgba_np_linear_f32=color_corrected_texture_rgba_np_linear_f32,
        floor_placement_descriptor=floor_placement_descriptor,
    )
    
    return answer
