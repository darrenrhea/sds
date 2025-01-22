import numpy as np
import rodrigues_utils


def nuke_world_to_undistorted_normalized_coordinates(p_giwc, camera_parameters_dict):
    """
    It is important to not only returns undistorted / ideal
    normalized x and y pixel coordinate, but also a boolean flag is_visible.
    because lens distortion makes things go pretty crazy off screen.
    TODO: remove nuke from the name, it has nothing to do with Nuke
    or any other lens distortion model.
    """
    assert isinstance(p_giwc, np.ndarray)

    camera_location = np.array(camera_parameters_dict["loc"])
    focal_length = camera_parameters_dict["f"]

    rodrigues_vector = np.array(camera_parameters_dict["rod"])

    world_to_camera = rodrigues_utils.SO3_from_rodrigues(rodrigues_vector)

    p_gicc = np.dot(world_to_camera, p_giwc - camera_location)

    x_over_z = p_gicc[0] / p_gicc[2]
    y_over_z = p_gicc[1] / p_gicc[2]
    x_normalized = x_over_z * focal_length
    y_normalized = y_over_z * focal_length

    is_visible = abs(x_normalized) <= 1.1 and abs(y_normalized) <= 9 / 16 * 1.1

    if not is_visible:
        return -1, -1, is_visible

    return x_normalized, y_normalized, is_visible

