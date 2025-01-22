from radial_polynomial_map import (
     radial_polynomial_map
)
import numpy as np
from CameraParameters import CameraParameters


def nuke_get_world_coordinates_of_pixel_coordinates_assuming_plane(
    x_pixel,
    y_pixel,
    camera_pose: CameraParameters,
    photograph_width_in_pixels,
    photograph_height_in_pixels,
    plane_coeffs,
    verbose=False
):
    """
    Calculates of the 3D world coordinated point (x, y, z), if any, that:
    1. is inside the given plane ax + by + cz + d = 0 where plane_coeffs = [a, b, c, d]
    2. would show up at that pixel location in the image.
    """
    if verbose:
        print("\n\nHello from nuke_get_world_coordinates_of_pixel_coordinates_assuming_plane")
        print(f"{x_pixel=}")
        print(f"{y_pixel=}")
        print(f"{photograph_width_in_pixels=}")
        print(f"{photograph_height_in_pixels=}")
        print(f"{plane_coeffs=}")
    
    focal_length = camera_pose.f
    k1 = camera_pose.k1
    k2 = camera_pose.k2
    k3 = camera_pose.k3
    p1 = camera_pose.p1
    p2 = camera_pose.p2
    ppi = camera_pose.ppi
    ppj = camera_pose.ppj


    world_to_camera = camera_pose.world_to_camera
    camera_to_world = world_to_camera.T

    distorted_x = (
          (x_pixel - (photograph_width_in_pixels/2))
        /
          (photograph_width_in_pixels/2)
    )   # should range from -1 to 1 for landscape 

    if verbose:
        print(f"from {x_pixel} we subtract  {photograph_width_in_pixels / 2} then divide by {photograph_width_in_pixels / 2} to get {distorted_x}")

    distorted_y = (
          (y_pixel - (photograph_height_in_pixels / 2))
        / 
          (photograph_width_in_pixels/2)
    )
    if verbose:
        print(f"{distorted_x=}")
        print(f"{distorted_y=}")

    distorted_x -= ppj
    distorted_y -= ppi

    if verbose:
        print(f"{distorted_x=}")
        print(f"{distorted_y=}")

    xp = distorted_x / focal_length 
    yp = distorted_y / focal_length

    if verbose:
        print(f"{xp=}")
        print(f"{yp=}")
    
    a, b = radial_polynomial_map(
        k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
        x=xp, y=yp
    )
    if verbose:
        print(f"{a=}")
        print(f"{b=}")


    x_normalized = a * focal_length
    y_normalized = b * focal_length

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
        1, # focal_length
        ],
        dtype=np.double
    )

    camera_cc = np.zeros((3,), dtype=np.double)  # of course the camera is at zero in its own coordinate system
     
    v_cc = screen_cc - camera_cc
    v_cc /= np.linalg.norm(v_cc)

    v_wc = np.dot(camera_to_world, v_cc)
    
    n = plane_coeffs[:3]
    n_dot_point_in_plane = - plane_coeffs[3]

    n_dot_loc = np.dot(camera_pose.loc, n)

    # find the t such that p_t = t*v_wc + camera_pose.loc
    # zeros out the plane equation n dot p =  n * point_in_plane
    # t* (n dot v_wc) + n dot loc = n dot point_in_plane
    # t = (n dot point_in_plane - n dot camera_loc_in_world_coordinates) / (n dot v_wc)

    t_hit = (n_dot_point_in_plane - n_dot_loc) / np.dot(n, v_wc)

    if t_hit > 0.0:
        x_hit = camera_pose.loc[0] + t_hit * v_wc[0]
        y_hit = camera_pose.loc[1] + t_hit * v_wc[1]
        z_hit = camera_pose.loc[2] + t_hit * v_wc[2]
        return x_hit, y_hit, z_hit
    else:
        print("No hit, i.e. you would not see that plane.")
        return None
