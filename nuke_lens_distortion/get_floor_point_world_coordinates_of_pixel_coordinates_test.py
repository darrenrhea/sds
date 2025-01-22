from nuke_world_to_pixel_coordinates import (
     nuke_world_to_pixel_coordinates
)

import numpy as np

from get_floor_point_world_coordinates_of_pixel_coordinates import (
     get_floor_point_world_coordinates_of_pixel_coordinates
)

from CameraParameters import CameraParameters



def test_get_floor_point_world_coordinates_of_pixel_coordinates_1():
    photograph_width_in_pixels = 1920
    photograph_height_in_pixels = 1080
    camera_parameters = CameraParameters(
        rod=[1.78669, -0.24683, 0.18642],
        loc=[0.24811, -124.06743, 34.55519],
        f=5.3835,
        ppi=0.1493,
        ppj=-0.02348,
        k1=-0.03657,
        k2=0.00413,
        k3=0.0,
        p1=0.00,
        p2=0.00
    )
   
    xy = nuke_world_to_pixel_coordinates(
        p_giwc=np.array([47.0, 25.0, 0.0]),
        camera_parameters=camera_parameters,
        photograph_width_in_pixels=photograph_width_in_pixels,
        photograph_height_in_pixels=photograph_height_in_pixels,
        verbose=True,
    )
    print(f"{xy=}")

    x_pixel = xy[0]
    y_pixel = xy[1]

    ans = get_floor_point_world_coordinates_of_pixel_coordinates(
        x_pixel=x_pixel,
        y_pixel=y_pixel,
        camera_parameters=camera_parameters,
        photograph_width_in_pixels=photograph_width_in_pixels,
        photograph_height_in_pixels=photograph_height_in_pixels,
        verbose=True,
    )
    assert abs(ans[0] - 47.0) < 1e-9
    assert abs(ans[1] - 25.0) < 1e-9
    print("test_get_floor_point_world_coordinates_of_pixel_coordinates passed")



if __name__ == "__main__":
    test_get_floor_point_world_coordinates_of_pixel_coordinates_1()
    