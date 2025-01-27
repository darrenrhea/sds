from prii_linear_f32 import (
     prii_linear_f32
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from get_a_random_floor_texture_for_this_context import (
     get_a_random_floor_texture_for_this_context
)
from augment_floor_texture_via_random_shadows_and_lights import (
     augment_floor_texture_via_random_shadows_and_lights
)


def test_augment_floor_texture_via_random_shadows_and_lights_1():
    floor_id = "24-25_HOU_CORE"

    floor_texture = get_a_random_floor_texture_for_this_context(
        floor_id=floor_id,
    )

    floor_texture_augmented_with_lights_and_shadows = (
        augment_floor_texture_via_random_shadows_and_lights(
            floor_texture=floor_texture,
            run_as_demo=True,
        )
    )
    
    
    # should it have points still on there?
    
    color_corrected_texture_rgba_np_linear_f32 = (
        floor_texture_augmented_with_lights_and_shadows[
            "color_corrected_texture_rgba_np_linear_f32"
        ]
    )
    
    prii_linear_f32(
        color_corrected_texture_rgba_np_linear_f32
    )
    # points = floor_texture_augmented_with_lights_and_shadows["points"]
    # prii_named_xy_points_on_image(
    #     name_to_xy=points,
    #     image=color_corrected_texture_rgba_np_linear_f32,
    #     output_image_file_path=None,
    #     default_color=(0, 255, 0),  # green is the default
    #     dont_show=False,
    # )





if __name__ == "__main__":
    test_augment_floor_texture_via_random_shadows_and_lights_1()