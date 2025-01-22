import numpy as np
from CameraParameters import CameraParameters
from nuke_lens_distortion import nuke_world_to_pixel_coordinates
from rodrigues_utils import SO3_from_rodrigues
from homography_utils import (
    get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters,
    get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel,
    map_points_through_homography,
    get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel_only_one_focal_length
)

def test_get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters():
    camera_parameters = CameraParameters(
        rod=[1.71147, 0.15901, -0.15101],
        loc=[-3.13258, -78.87846, 17.57712],
        f=2.8293,
        ppi=0,
        ppj=0,
        k1=0,
        k2=0,
        k3=0,
        p1=0,
        p2=0
    )

    print(camera_parameters)
    camera_parameters.fx = camera_parameters.f
    camera_parameters.fy = camera_parameters.f
    world_floor_coordinates_to_photo_pixel_homography = get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters(
       camera_parameters
    )

    print(
        "The world_floor_coordinates_to_photo_pixel_homography is H=\n",
        world_floor_coordinates_to_photo_pixel_homography
    )

    # make some random locations in the world floor:
    xys = 5 * np.random.randn(10, 2)

    pixels = map_points_through_homography(
        xys,
        homography=world_floor_coordinates_to_photo_pixel_homography
    )

    print(pixels)
    should_be = np.zeros_like(pixels)

    for k, xy in enumerate(xys):
        xyz = np.array([xy[0], xy[1], 0])
        ans = nuke_world_to_pixel_coordinates(
            p_giwc=xyz,
            camera_parameters=camera_parameters,
            photograph_width_in_pixels=1920,
            photograph_height_in_pixels=1080
        )
        print(ans)
        should_be[k, 0] = ans[0]
        should_be[k, 1] = ans[1]
    assert np.allclose(should_be, pixels)


def test_get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel():
    np.set_printoptions(precision=15)
    actual_rod = [1.719876543210123, 0.159876543210123, -0.1510123456789]
    actual_loc = [-3.1329876543210123, -78.9876543210123, 17.98765432101234]
    actual_fx = 0.1 + np.random.rand()
    actual_fy = 0.1 + np.random.rand()
    camera_parameters = CameraParameters(
        rod=actual_rod,
        loc=actual_loc,
        f=0,   # going to assign two focal lengths, fx, fy separately
        ppi=0,
        ppj=0,
        k1=0,
        k2=0,
        k3=0,
        p1=0,
        p2=0
    )
    camera_parameters.fx = actual_fx
    camera_parameters.fy = actual_fy

    print(f"actual_fx={camera_parameters.fx}")
    print(f"actual_fy={camera_parameters.fy}")

    print("actual_R=\n", SO3_from_rodrigues(camera_parameters.rod))

    world_floor_coordinates_to_photo_pixel_homography = get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters(
       camera_parameters
    )

    print(
        "The world_floor_coordinates_to_photo_pixel_homography is H=\n",
        world_floor_coordinates_to_photo_pixel_homography
    )

    while True:
        s = np.random.randn()
        if abs(s) > 0.1:  # multiplying the homography by near-zero is not okay.
            break

    recovered = get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel(
        s * world_floor_coordinates_to_photo_pixel_homography
    )

    assert np.allclose(recovered["rod"], actual_rod)
    assert np.allclose(recovered["loc"], actual_loc)
    assert np.allclose(recovered["fx"], actual_fx)
    assert np.allclose(recovered["fy"], actual_fy)


def test2():
    homography_from_photo_to_world_floor = np.array([[ 9.04276511e-01, -2.12937328e-01,  2.58168961e+02],
       [-3.72384781e-01, -4.56139156e+00,  3.00778685e+03],
       [-8.32399032e-04,  5.66090895e-02,  4.23475688e-01]])

    cp = get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel(
        np.linalg.inv(homography_from_photo_to_world_floor)
    )
    


def test_get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel_only_one_focal_length():
    np.set_printoptions(precision=15)
    actual_rod = [1.719876543210123, 0.159876543210123, -0.1510123456789]
    actual_loc = [-3.1329876543210123, -78.9876543210123, 17.98765432101234]
    actual_fx = 0.1 + np.random.rand()
    actual_fy = actual_fx
    camera_parameters = CameraParameters(
        rod=actual_rod,
        loc=actual_loc,
        f=0,   # going to assign two focal lengths, fx, fy separately
        ppi=0,
        ppj=0,
        k1=0,
        k2=0,
        k3=0,
        p1=0,
        p2=0
    )
    camera_parameters.fx = actual_fx
    camera_parameters.fy = actual_fy

    print(f"actual_fx={camera_parameters.fx}")
    print(f"actual_fy={camera_parameters.fy}")

    print("actual_R=\n", SO3_from_rodrigues(camera_parameters.rod))

    world_floor_coordinates_to_photo_pixel_homography = get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters(
       camera_parameters
    )

    print(
        "The world_floor_coordinates_to_photo_pixel_homography is H=\n",
        world_floor_coordinates_to_photo_pixel_homography
    )

    while True:
        s = np.random.randn()
        if abs(s) > 0.1:  # multiplying the homography by near-zero is not okay.
            break

    recovered = get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel_only_one_focal_length(
        s * world_floor_coordinates_to_photo_pixel_homography
    )

    assert np.allclose(recovered["rod"], actual_rod)
    assert np.allclose(recovered["loc"], actual_loc)
    assert np.allclose(recovered["fx"], actual_fx)
    assert np.allclose(recovered["fy"], actual_fy)



def test_get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel_only_one_focal_length_under_noise():
    np.set_printoptions(precision=15)
    actual_rod = [1.719876543210123, 0.159876543210123, -0.1510123456789]
    actual_loc = [-3.1329876543210123, -78.9876543210123, 17.98765432101234]
    actual_fx = 0.1 + 3 * np.random.rand()
    actual_fy = actual_fx
    camera_parameters = CameraParameters(
        rod=actual_rod,
        loc=actual_loc,
        f=0,   # going to assign two focal lengths, fx, fy separately
        ppi=0,
        ppj=0,
        k1=0,
        k2=0,
        k3=0,
        p1=0,
        p2=0
    )
    camera_parameters.fx = actual_fx
    camera_parameters.fy = actual_fy

    print(f"actual_fx={camera_parameters.fx}")
    print(f"actual_fy={camera_parameters.fy}")
    print("actual_loc=\n", camera_parameters.loc)
    print("actual_R=\n", SO3_from_rodrigues(camera_parameters.rod))

    world_floor_coordinates_to_photo_pixel_homography = get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters(
       camera_parameters
    )
    world_floor_coordinates_to_photo_pixel_homography /= world_floor_coordinates_to_photo_pixel_homography[2,2]
    print(world_floor_coordinates_to_photo_pixel_homography)

    world_floor_coordinates_to_photo_pixel_homography += np.random.randn(3,3) * 0.001

    print(
        "The world_floor_coordinates_to_photo_pixel_homography is H=\n",
        world_floor_coordinates_to_photo_pixel_homography
    )

    while True:
        s = np.random.randn()
        if abs(s) > 0.1:  # multiplying the homography by near-zero is not okay.
            break

    recovered = get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel_only_one_focal_length(
        s * world_floor_coordinates_to_photo_pixel_homography
    )

    recovered_cp = CameraParameters(
        rod=np.array(recovered["rod"]),
        loc=np.array(recovered["loc"]),
        f=recovered["fx"]
    )
    recovered_cp.fx = recovered["fx"]
    recovered_cp.fy = recovered["fy"]

    recovered_homography = get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters(
       recovered_cp
    )
    recovered_homography /= recovered_homography[2,2]
    print(recovered_homography)
    recovered_R = SO3_from_rodrigues(recovered_cp.rod)
    print(f"recovered_R=\n{recovered_R}")
    print(f"recovered_loc=\n{recovered_cp.loc}")
    
    # assert np.allclose(recovered["rod"], actual_rod)
    # assert np.allclose(recovered["loc"], actual_loc)
    # assert np.allclose(recovered["fx"], actual_fx)
    # assert np.allclose(recovered["fy"], actual_fy)



if __name__ == "__main__":
    # test_get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters()
    # test_get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel()
    # test_get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel_only_one_focal_length()
    test_get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel_only_one_focal_length_under_noise()
