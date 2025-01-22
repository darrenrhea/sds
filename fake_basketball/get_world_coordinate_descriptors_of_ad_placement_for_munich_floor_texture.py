from typing import List

from AdPlacementDescriptor import (
     AdPlacementDescriptor
)

def get_world_coordinate_descriptors_of_ad_placement_for_munich_floor_texture() -> List[AdPlacementDescriptor]:
    """
    We guess guessed and checked until the London floor texture completely covers the Munich floor.
    """

    ad_descriptors = []
   
    # the legal playing area is 28 meters wide x 15 meters high
    # the texture may not be aspect ratio preserving
    # geometry = get_euroleague_geometry()
    # points = geometry["points"]
    # left_x_margin = 233  * 28 / 3373 * 1.166
    # right_x_margin = 233 * 28 / 3373 * 1.166
    # top_y_margin = 294 * 15 / 1810 * 0.85
    # bottom_y_margin = 296 * 15 / 1810

    # x_min = -14.0 - left_x_margin
    # x_max = 14.0 + right_x_margin
    # y_min = -7.5 - bottom_y_margin
    # y_max = 7.5 + top_y_margin
    x_min=-16.255257634153573
    x_max=16.255257634153573
    y_min=-9.953038674033149 - 0.75
    y_max=9.570994475138122

    descriptor = AdPlacementDescriptor(
        name="floor_texture",
        origin=[x_min, y_min, 0.0],
        u=[1.0, 0.0, 0.0],
        v=[0.0, 1.0, 0.0],
        height=y_max - y_min,
        width=x_max - x_min,
    )
    ad_descriptors.append(descriptor)
    assert len(ad_descriptors) == 1
    return ad_descriptors    
            
