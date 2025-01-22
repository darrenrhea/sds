
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
from CameraParameters import CameraParameters
import numpy as np


def get_t_hit(
    ad_placement_descriptor: AdPlacementDescriptor,
    camera_parameters: CameraParameters,
    vx_wc: np.array,
    vy_wc: np.array,
    vz_wc: np.array,
):
    """
    For each ad, get the hitting time, if any, of the "backwards light ray"
    from camera to ad.  Hitting time should be positive if the object/ad is visible.
    """
   
   
    u = ad_placement_descriptor.u
    v = ad_placement_descriptor.v
    ad_origin = ad_placement_descriptor.origin
    cameras_location_in_world_coordinates = camera_parameters.loc

    # Suppose we are hitting the plane spanned by u, v starting at "ad_origin":
    # normal = np.cross(u, v)
    # plane is the locus of x such that np.dot(x - origin, normal) = 0
    # x dot normal = origin dot normal
    # so we want to solve for t the equation np.dot(cameras_location_in_world_coordinates + t * velocity_wc, normal) =  origin dot normal
    # so t = (<origin | normal> - <cameras_location_in_world_coordinates| normal>) / <velocity_wc| normal>
    
    n = np.cross(u, v)
    n /= np.linalg.norm(n)
    # print("Internal to get_t_hit")
    # print(f"{u=}")
    # print(f"{v=}")
    # print(f"{n=}")
    normal_x, normal_y, normal_z = n


    numerator = np.dot(ad_origin - cameras_location_in_world_coordinates, n)
    
    assert isinstance(numerator, float), f"numerator should be a float but is {type(numerator)=}"

    denom = vx_wc * normal_x + vy_wc * normal_y + vz_wc * normal_z

    t_hit = numerator / denom
    assert t_hit.shape == vx_wc.shape, f"{t_hit.shape=} {vx_wc.shape=}"
    return t_hit
