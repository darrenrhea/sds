from augment_the_floor_texture_for_bal import (
     augment_the_floor_texture_for_bal
)
from augment_the_floor_texture_for_allstars import (
     augment_the_floor_texture_for_allstars
)
from add_shadows import (
     add_shadows
)
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
import numpy as np
from prii_linear_f32 import (
     prii_linear_f32
)
from add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32 import (
     add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32
)


def augment_floor_texture_via_random_shadows_and_lights(
    floor_texture: dict,
    run_as_demo: bool = False,
):
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
    assert isinstance(floor_texture, dict)
    assert "color_corrected_texture_rgba_np_linear_f32" in floor_texture
    assert "floor_placement_descriptor" in floor_texture
   
    color_corrected_texture_rgba_np_linear_f32 = floor_texture["color_corrected_texture_rgba_np_linear_f32"]
    floor_placement_descriptor = floor_texture["floor_placement_descriptor"]

    if run_as_demo:
        print("this is the original floor texture before adding shadows and lights")
        prii_linear_f32(
            color_corrected_texture_rgba_np_linear_f32
        )
        
        print(
            floor_placement_descriptor
        )

    color_corrected_texture_rgb_np_linear_f32 = color_corrected_texture_rgba_np_linear_f32[:, :, :3]
    
    # augmented_rgb_np_linear_f32 = augment_the_floor_texture_for_allstars(
    #     rgb_np_linear_f32=color_corrected_texture_rgb_np_linear_f32
    # )

    augmented_rgb_np_linear_f32 = augment_the_floor_texture_for_bal(
        rgb_np_linear_f32=color_corrected_texture_rgb_np_linear_f32
    )

    with_shadows_rgb_np_linear_f32 = add_shadows(
        augmented_rgb_np_linear_f32
    )

    if run_as_demo:
        prii_linear_f32(
            x=with_shadows_rgb_np_linear_f32,
            caption="this is the shadows-and-lights-added floor texture",
        )
    
    final_rgba_np_linear_f32 = add_an_opaque_alpha_channel_to_create_rgba_hwc_np_f32(
        with_shadows_rgb_np_linear_f32
    )
    
    assert final_rgba_np_linear_f32.shape[2] == 4
    assert final_rgba_np_linear_f32.dtype == "float32"
    assert isinstance(floor_placement_descriptor, AdPlacementDescriptor)

    floor_texture_augmented_with_lights_and_shadows = dict(
        color_corrected_texture_rgba_np_linear_f32=final_rgba_np_linear_f32,
        floor_placement_descriptor=floor_placement_descriptor,
    )
    
    return floor_texture_augmented_with_lights_and_shadows


