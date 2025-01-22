"""
In the Nuke lens distortion model, the map from
distorted normalized screen coordinates in [-1, 1] x [-9/16, 9/16] (or whatever your aspect ratio is)
to undistorted screen coordinates is given by a polynomial
(is closed form and simple), whereas the other direction,
from undistorted to distorted, is not closed form and thus has to be solved
by Newton's method etc.
"""
from camera2floor import (
     camera2floor
)
from convert_old_camera_parameters_format_to_new_camera_parameters_format import (
     convert_old_camera_parameters_format_to_new_camera_parameters_format
)
from get_floor_point_world_coordinates_of_pixel_coordinates import (
     get_floor_point_world_coordinates_of_pixel_coordinates
)
from inverse_of_radial_polynomial_map import (
     inverse_of_radial_polynomial_map
)
from nuke_world_to_pixel_coordinates import (
     nuke_world_to_pixel_coordinates
)
from nuke_world_to_undistorted_normalized_coordinates import (
     nuke_world_to_undistorted_normalized_coordinates
)
from radial_polynomial_map import (
     radial_polynomial_map
)

__all__ = [
    "camera2floor",
    "convert_old_camera_parameters_format_to_new_camera_parameters_format",
    "get_floor_point_world_coordinates_of_pixel_coordinates",
    "inverse_of_radial_polynomial_map",
    "nuke_world_to_pixel_coordinates",
    "nuke_world_to_undistorted_normalized_coordinates",
    "radial_polynomial_map",
]
