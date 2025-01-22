from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)

import numpy as np


def get_led_corners(
    court_id
):    
    """
    How should we ask for the led board corners?
    """
    dct = {}

    court_id_to_clip_id = {
        "ThomasMack": "slgame1",
    }
        
    clip_id = court_id_to_clip_id[court_id]

    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.0
    )

    for led_board_index, ad_placement_descriptor in enumerate(ad_placement_descriptors):
        board_name = f"LED{led_board_index}"
        assert isinstance(ad_placement_descriptor, AdPlacementDescriptor)
        dct[f"{board_name}_tl"] = list(ad_placement_descriptor.tl())
        dct[f"{board_name}_tr"] = list(ad_placement_descriptor.tr())
        dct[f"{board_name}_bl"] = list(ad_placement_descriptor.bl())
        dct[f"{board_name}_br"] = list(ad_placement_descriptor.br())

    return dct


if __name__ == "__main__":
    print(get_led_corners("ThomasMack"))