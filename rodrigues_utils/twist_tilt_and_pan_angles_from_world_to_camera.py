import numpy as np


def twist_tilt_and_pan_angles_from_world_to_camera(
    world_to_camera: np.ndarray
):  
    """
    These three angles are more human understandable
    than the rodrigues vector.
    """
    assert world_to_camera.shape == (3, 3)
    assert np.isclose(np.linalg.det(world_to_camera), 1)  # the special in SO_3
    assert np.allclose(world_to_camera @ world_to_camera.T, np.eye(3))  # the orthogonal in SO_3

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
    the_cameras_z_axis = world_to_camera[2]
    the_cameras_z_axis_x_component = the_cameras_z_axis[0]
    the_cameras_z_axis_y_component = the_cameras_z_axis[1]
    # only panning, not twisting nor tilting, changes the angle of (the cameras_z_axis's projection into the world z-plane)
    theta_z = np.arctan(the_cameras_z_axis_x_component / the_cameras_z_axis_y_component)
    pan_angle_in_degrees = theta_z * 180 / np.pi
    # print(f"We think pan_angle_in_degrees = {pan_angle_in_degrees}")
    rot_around_world_z_axis = np.array(
        [
            [np.cos(theta_z), np.sin(theta_z), 0],
            [-np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )
    
    # only tilting, not panning nor twisting, moves camera_z_axis up and down:
    the_cameras_z_axis_z_component = the_cameras_z_axis[2]
    theta_x = np.arcsin(the_cameras_z_axis_z_component)
    tilt_angle_in_degrees = theta_x * 180 / np.pi
    # print(f"We think tilt_angle_in_degrees = {tilt_angle_in_degrees}")
    rot_around_world_x_axis = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )

    # world_to_camera = the_cameras_x_y_and_z_axes_as_columns_giwc_prior_to_yaw_tilt_and_pan.T @ rot_around_world_y_axis.T @ rot_around_world_x_axis.T @ rot_around_world_z_axis.T 

    rot_around_world_y_axis_T =  the_cameras_x_y_and_z_axes_as_columns_giwc_prior_to_yaw_tilt_and_pan @ world_to_camera @ rot_around_world_z_axis @ rot_around_world_x_axis
    rot_around_world_y_axis = rot_around_world_y_axis_T.T

    theta_y = np.arctan(rot_around_world_y_axis[0,2] / rot_around_world_y_axis[0,0])
     
    twist_angle_in_degrees = theta_y * 180 / np.pi
    # print(f"We think twist_angle_in_degrees = {twist_angle_in_degrees}")


    rot_around_world_y_axis = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ]
    )
    return (
        twist_angle_in_degrees,
        tilt_angle_in_degrees,
        pan_angle_in_degrees
    )
    

