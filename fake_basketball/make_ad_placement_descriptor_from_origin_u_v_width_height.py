import numpy as np

from AdPlacementDescriptor import (
     AdPlacementDescriptor
)


def make_ad_placement_descriptor_from_origin_u_v_width_height(
    origin,
    u,
    v,
    width,
    height,
    overcover_by,
):
    assert isinstance(width, float), f"width {width}"
    assert isinstance(height, float)

    origin = np.array(
        origin,
        dtype=np.float64
    )

    u = np.array(
        u,
        dtype=np.float64
    )

    v = np.array(
        v,
        dtype=np.float64
    )  

    half_width = width / 2
    half_height = height / 2

    bl = origin - u * half_width - v * half_height
    tl = origin - u * half_width + v * half_height
    br = origin + u * half_width - v * half_height
    


    if overcover_by > 0:
        delta_x = overcover_by
        delta_z = overcover_by
        tl += - delta_x * u + delta_z * v
        bl += - delta_x * u - delta_z * v
        br +=   delta_x * u - delta_z * v

    ad_placement_descriptor = AdPlacementDescriptor(
        name="LED",  # so far only one LED board in NBA
        origin=bl,
        u=u,
        v=v,
        height=np.linalg.norm(tl - bl),
        width=np.linalg.norm(br - bl) * 1.00 #
    )

    return ad_placement_descriptor
