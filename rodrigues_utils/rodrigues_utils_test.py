import numpy as np
from rodrigues_utils import *
from rodrigues_utils import my_SO3_from_rodrigues, rotation_matrix_from_twist_tilt_and_pan, twist_tilt_and_pan_angles_from_world_to_camera

np.set_printoptions(precision=15)

def normalized(v):
    return v / np.linalg.norm(v)


def test_rodrigues_from_SO3():
    random_angle = np.pi * np.random.rand()  # make a random angle between 0 and pi
    random_unit_vector = normalized(np.random.randn(3))  # make a random unit vector
    original_rodrigues = random_unit_vector * random_angle  # a random angle-axis vector
    # original_rodrigues = np.array([-2.5156862758, -0.2337204158, -1.8619759948])

    # print(np.linalg.norm(original_rodrigues))
    # print(f"original={original_rodrigues}")
    rotation_matrix = SO3_from_rodrigues(original_rodrigues)
    # print(rotation_matrix)
    recovered_rodrigues = rodrigues_from_SO3(rotation_matrix)
    # print(f"recovered={recovered_rodrigues}")
    assert np.allclose(recovered_rodrigues, original_rodrigues)
    # print("passed")


def test_canonicalize_rodrigues():
    bad_rodrigues = np.array([3.4345609132, -2.5786765607, -7.2057107968])
    canonical = canonicalize_rodrigues(bad_rodrigues)
    # print(canonical)
    should_be = np.array([0.86201381, -0.64720204, -1.80850548])
    assert np.allclose(canonical, should_be)


def test_two_implementations_agree():
    r = make_random_rodrigues_vector()
    r = np.array([np.pi /2 , 0, 0])
    A = SO3_from_rodrigues(r)
    B = my_SO3_from_rodrigues(r)
    assert np.allclose(A, B)


def test_rotation_matrix_from_twist_tilt_and_pan():
    # world_to_camera can be thought of as 3 rows, which are the camera's three axes expressed in world coordinates:
    twist_actually_is = rotation_matrix_from_twist_tilt_and_pan(
        twist_angle_in_degrees=45,
        tilt_angle_in_degrees=0,
        pan_angle_in_degrees=0
    )
    # If you are the operator behind the camera, grab the camera and twist it "clockwise" about the optical axis.
    twist_should_be = np.array(
        [
            [ 0.707106781186548,  0.               , -0.707106781186547],  # camera's x axis is now this in world coordinates
            [-0.707106781186547,  0.               , -0.707106781186548],  # camera's y axis is now this in world coordinates
            [ 0.               ,  1.               ,  0.               ],  # camera's z axis is still this in world coordinates
        ]
    )
    assert np.allclose(twist_should_be, twist_actually_is)

    tilt_actually_is = rotation_matrix_from_twist_tilt_and_pan(
        twist_angle_in_degrees=0,
        tilt_angle_in_degrees=45,  # points the camera 45 degrees up into the sky / right-hand-rule around camera's x axis
        pan_angle_in_degrees=0
    )
    tilt_should_be = np.array(
        [
            [ 1.               ,  0.               ,  0.               ],
            [ 0.               ,  0.707106781186547, -0.707106781186548],
            [ 0.               ,  0.707106781186548,  0.707106781186547]
        ]
    )
    
    assert np.allclose(tilt_should_be, tilt_actually_is)

    pan_actually_is = rotation_matrix_from_twist_tilt_and_pan(
        twist_angle_in_degrees=0,
        tilt_angle_in_degrees=0,
        pan_angle_in_degrees=45
    )

    pan_should_be = np.array(
        [
            [ 0.707106781186548, -0.707106781186547,  0.               ],
            [ 0.               ,  0.               , -1.               ],
            [ 0.707106781186547,  0.707106781186548,  0.               ]
        ]
    )
    assert np.allclose(pan_should_be, pan_actually_is)
    

    

def test_twist_tilt_and_pan_angles_from_world_to_camera():
    # world_to_camera can be thought of as 3 rows, which are the camera's three axes expressed in world coordinates:
    num_trials = 1000
    for _ in range(num_trials):
        twist_should_be = np.random.randint(-89, 89)
        tilt_should_be = np.random.randint(-89, 89)
        pan_should_be = np.random.randint(-89, 89)
        pan_only = rotation_matrix_from_twist_tilt_and_pan(
            twist_angle_in_degrees=twist_should_be,
            tilt_angle_in_degrees=tilt_should_be,
            pan_angle_in_degrees=pan_should_be
        )

        twist, tilt, pan = twist_tilt_and_pan_angles_from_world_to_camera(pan_only)
        assert abs(twist - twist_should_be) < 1e-13
        assert abs(tilt - tilt_should_be) < 1e-13
        assert abs(pan - pan_should_be) < 1e-13

    

    
    

if __name__ == "__main__":
    # test_rodrigues_from_SO3()
    # test_canonicalize_rodrigues()
    # test_two_implementations_agree()
    # test_rotation_matrix_from_twist_tilt_and_pan()
    # test_twist_tilt_and_pan_angles_from_world_to_camera()

    original_rodrigues = np.array([1.8475977, -0.1597331, 0.113842264 ])
    rotation_matrix = SO3_from_rodrigues(original_rodrigues)
    print(rotation_matrix)