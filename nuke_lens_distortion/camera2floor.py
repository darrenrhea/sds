from radial_polynomial_map import (
     radial_polynomial_map
)
import numpy as np


# CHAZ: this is just get_floor_point_world_coordinates_of_pixel_coordinates but alrready given the normalized (x,y) camera pixel coordinates
def camera2floor(
    distorted_x,
    distorted_y,
    camera_parameters,
):
    """
    Rename this function, it is wrong!
    Calculate where in the floor {z=0} would show up at that pixel location.
    """
    focal_length = camera_parameters.f
    
    k1 = camera_parameters.k1
    k2 = camera_parameters.k2
    k3 = camera_parameters.k3
    p1 = camera_parameters.p1
    p2 = camera_parameters.p2
    ppi = camera_parameters.ppi
    ppj = camera_parameters.ppj


    world_to_camera = camera_parameters.world_to_camera
    camera_to_world = world_to_camera.T

    """
    distorted_x = (
          (x_pixel - (photograph_width_in_pixels/2))
        /
          (photograph_width_in_pixels/2)
    )   # should range from -1 to 1 for landscape 
    distorted_y = (
          (y_pixel - (photograph_height_in_pixels / 2))
        / 
          (photograph_width_in_pixels/2)
    )
    """
    distorted_x -= ppj
    distorted_y -= ppi


    """
    distorted_x, distorted_y = inverse_of_radial_polynomial_map(
        k1=cp.k1, k2=cp.k2, k3=cp.k3,
        a=x_normalized - cp.ppj,
        b=y_normalized - cp.ppi
    )
    """

    a, b = radial_polynomial_map(
        k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
        x=distorted_x, y=distorted_y
        
    )

    """
    a=x_normalized - cp.ppj
    b=y_normalized - cp.ppi
    """

    x_normalized = a
    y_normalized = b

    screen_cc = np.array(
        [
        x_normalized,
        y_normalized,
        focal_length
        ],
        dtype=np.double
    )

    camera_cc = np.zeros((3,), dtype=np.double)  # of course the camera is at zero in its own coordinate system
     
    v_cc = screen_cc - camera_cc
    v_cc /= np.linalg.norm(v_cc)

    v_wc = np.dot(camera_to_world, v_cc)
    camera_location_z = camera_parameters.loc[2]
    t_hit = (0.0 - camera_location_z) / v_wc[2]

    if t_hit > 0.0:
        x_hit = camera_parameters.loc[0] + t_hit * v_wc[0]
        y_hit = camera_parameters.loc[1] + t_hit * v_wc[1]
        return x_hit, y_hit, 0.0
    else:
        return None