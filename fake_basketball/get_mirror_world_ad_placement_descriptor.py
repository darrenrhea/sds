from typing import List
import numpy as np

from AdPlacementDescriptor import (
     AdPlacementDescriptor
)


def get_mirror_world_ad_placement_descriptor(
    clip_id: str,
    with_floor_as_giant_ad: bool,
    overcover_by: float = 0.0  # sometimes we want the ad to be much bigger behind its hole, or for relevance masks that are slightly bigger.
) -> List[AdPlacementDescriptor]:
    """
    We need to act like the LED board is a mirror sometimes.
    """
    # can we really determine the world coordinates of the LED board corners from the context_id alone?
    # this is assuming things don't change between games!
    clip_id_to_context_id = {
        "munich2024-01-25-1080i-yadif": "munich2024",
        "munich2024-01-09-1080i-yadif": "munich2024",
        "bay-zal-2024-03-15-mxf-yadif": "munich2024",
        "bay-mta-2024-03-22-mxf": "munich2024",
        "bos-mia-2024-04-21-mxf": "boston_celtics",
        "bos-dal-2024-06-09-mxf": "boston_celtics",
        "dal-bos-2024-01-22-mxf": "dallas_mavericks",
        "dal-bos-2024-06-12-mxf": "dallas_mavericks",
        "slgame1": "summer_league_2024",
    }

    if clip_id not in clip_id_to_context_id:
        raise Exception(f"You need to say what the context_id of {clip_id=} is in {clip_id_to_context_id=}")

    context_id = clip_id_to_context_id[clip_id]

    if context_id == "boston_celtics":
        led_screen_y_coordinate = 30.339 + 1.0


        origin = np.array([-56.0, led_screen_y_coordinate, -0.2])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        height = 30 - (-26)
        width = 56 - (-56)

        ad_placement_descriptor = AdPlacementDescriptor(
            name="mirror_world_floor",  # so far only one LED board in NBA
            origin=origin,
            u=u,
            v=v,
            height=height,
            width=width,
        )
    elif context_id == "dallas_mavericks":
        led_screen_y_coordinate = 30.339 + 1.0

        origin = np.array([-56.0, led_screen_y_coordinate, 0.0])
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        height = 30 - (-26)
        width = 56 - (-56)

        ad_placement_descriptor = AdPlacementDescriptor(
            name="mirror_world_floor",  # so far only one LED board in NBA
            origin=origin,
            u=u,
            v=v,
            height=height,
            width=width,
        )
    else:
        raise Exception(f"unknown context_id {context_id}")


    return ad_placement_descriptor   
            
