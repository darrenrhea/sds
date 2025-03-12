from pathlib import Path
from prii_linear_f32 import (
     prii_linear_f32
)
from augment_floor_texture_via_random_shadows_and_lights import (
     augment_floor_texture_via_random_shadows_and_lights
)
from get_a_random_floor_texture_for_this_context import (
     get_a_random_floor_texture_for_this_context
)


def get_a_floor_texture_with_random_shadows_and_lights(
    floor_id: str,
    asset_repos_dir: Path,
    verbose: bool = False,
) -> dict:
    """
    Grab a floor texture for this context and augment it with random shadows and lights.
    """

    floor_texture = get_a_random_floor_texture_for_this_context(
        floor_id=floor_id,
        asset_repos_dir=asset_repos_dir,
    )

    floor_texture_augmented_with_lights_and_shadows = augment_floor_texture_via_random_shadows_and_lights(
        floor_texture=floor_texture,
    )
    if verbose:
        color_corrected_texture_rgba_np_linear_f32 = (
            floor_texture_augmented_with_lights_and_shadows[
                "color_corrected_texture_rgba_np_linear_f32"
            ]
        )
        
        prii_linear_f32(
            color_corrected_texture_rgba_np_linear_f32
        )

    return floor_texture_augmented_with_lights_and_shadows

