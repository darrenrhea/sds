from inverse_of_radial_polynomial_map import (
     inverse_of_radial_polynomial_map
)
import numpy as np
from CameraParameters import (
     CameraParameters
)


def nuke_world_to_pixel_coordinates(
    p_giwc,
    camera_parameters,
    photograph_width_in_pixels,
    photograph_height_in_pixels,
    verbose=False
):
    """
    WARNING:  this is overly optimistic about the visibility of points.
    Points that are slightly off screen will be called visible even though they are not.
    
    Outputs DISTORTED pixel coordinates, which is why it is so expensive.
    It is important to not only return x and y pixel coordinates, but also a boolean flag is_visible
    because lens distortion makes things go pretty crazy off screen,
    and you should not trust it.
    """
    assert isinstance(camera_parameters, CameraParameters)
    assert isinstance(p_giwc, np.ndarray)
    assert p_giwc.shape == (3,)
    cp = camera_parameters
    p_gicc = np.dot(cp.world_to_camera, p_giwc - cp.loc)
    if verbose:
        print(f"{p_giwc=}")
    x_over_z = p_gicc[0] / p_gicc[2]
    y_over_z = p_gicc[1] / p_gicc[2]
    
   
    
    # do a crude calculation to see if the point is even remotely visible:
    x_normalized = x_over_z * cp.f + cp.ppj
    y_normalized = y_over_z * cp.f + cp.ppi

    # if verbose:
    #     print(f"\n{x_normalized=}")
    #     print(f"{y_normalized=}")   

    is_visible = abs(x_normalized) <= 1.1 and abs(y_normalized) <= photograph_height_in_pixels / photograph_width_in_pixels * 1.1

    if not is_visible:
        return -1, -1, is_visible

    if verbose:
        print(f"\n{x_over_z=}")
        print(f"{y_over_z=}")
        print("will go through the hard to calculate direction")
    
    distorted_x, distorted_y = inverse_of_radial_polynomial_map(
        k1=cp.k1, k2=cp.k2, k3=cp.k3, p1=cp.p1, p2=cp.p2,
        a=x_over_z, # x_normalized - cp.ppj,
        b=y_over_z, #y_normalized - cp.ppi
    )

    if verbose:
        print("gives:")
        print(f"x_distorted_on_focal_length_1_plane={distorted_x}")
        print(f"y_distorted_on_focal_length_1_plane={distorted_y}")

    distorted_x *= cp.f
    distorted_y *= cp.f

    if verbose:
        print(f"x_distorted={distorted_x}")
        print(f"y_distorted={distorted_y}")
  
    
    distorted_x += cp.ppj
    distorted_y += cp.ppi

    if verbose:
        print(f"x_distorted_post_principal_point={distorted_x}")
        print(f"y_distorted_post_principal_point={distorted_y}")
    
    pixels_x = (
        photograph_width_in_pixels / 2 + photograph_width_in_pixels / 2 * distorted_x
    )

    pixels_y = (
        photograph_height_in_pixels / 2 + photograph_width_in_pixels / 2 * distorted_y
    )
    if verbose:
        print(f"pixels_x={pixels_x}")
        print(f"pixels_y={pixels_y}")

    return pixels_x, pixels_y, is_visible
