

import numpy as np





def rotation_matrix_from_twist_tilt_and_pan(
    twist_angle_in_degrees,
    tilt_angle_in_degrees,
    pan_angle_in_degrees
):
    """ 
    If you start the camera pointing forward along the world-y-axis,
    as is typical in our basketball world coordinate convention,
    then twist the camera right-rule around its optical-axis = (the-camera's z-axis) = (the world-y-axis) (most people would call this clockwise) by twist_angle_in_degrees,
    then tilt downward to the earth by tilt_angle_in_degrees, then pan to the right by angle_in_degrees,
    what is world_to_camera, i.e. the camera's axes expressed in world coordinates as 3 rows?
    See the tests.
    """
    the_cameras_x_y_and_z_axes_as_rows_giwc_prior_to_yaw_tilt_and_pan = np.array(
        [
            [1, 0, 0],  # the camera's x axis giwc
            [0, 0, -1],  # the camera's y axis giwc
            [0, 1, 0],  # the camera's z axis giwc
        ]
    )
    the_cameras_x_y_and_z_axes_as_columns_giwc_prior_to_yaw_tilt_and_pan = (
        the_cameras_x_y_and_z_axes_as_rows_giwc_prior_to_yaw_tilt_and_pan.T
    )

    theta_y = twist_angle_in_degrees / 180 * np.pi
    theta_x = tilt_angle_in_degrees / 180 * np.pi
    theta_z = pan_angle_in_degrees / 180 * np.pi
    # positive twist
    rot_around_world_x_axis = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )
    rot_around_world_y_axis = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ]
    )

    rot_around_world_z_axis = np.array(
        [
            [np.cos(theta_z), np.sin(theta_z), 0],
            [-np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )
    
    world_to_camera = the_cameras_x_y_and_z_axes_as_columns_giwc_prior_to_yaw_tilt_and_pan.T @ rot_around_world_y_axis.T @ rot_around_world_x_axis.T @ rot_around_world_z_axis.T 
   
    np.allclose(np.dot(world_to_camera.T, world_to_camera), np.eye(3))  # assert it is in SO_3
    return world_to_camera


