import numpy as np
from homography_utils import solve_for_homography_based_on_4_point_correspondences, map_points_through_homography
from rodrigues_utils import SO3_from_rodrigues, rodrigues_from_SO3
from CameraParameters import CameraParameters

np.set_printoptions(precision=15, suppress=True)
# its a tripod, so both cameras share same location:
loc = np.array([0.0, 0.0, 0.0])



def convert_pixel_units_homography_to_normalized_units_homography(H_in_pixel_units):
    """
    If you wrote 2D-2D correspondences between two frames in the standard pixel coordinates,
    i.e. x growing rightward from 0 to 1919 followed by y growing downward from 0 to 1079,
    what would the homography be had it been in normalized coordinates,
    i.e. x ranges from -1 to 1 left to right, y ranges from -9/16 to 9/16 top to bottom?
    """
    from_normalized_to_pixel = np.array(
        [
            [1920/2,      0, 1920/2],
            [     0, 1920/2, 1080/2],
            [     0,      0,      1]
        ]
    )
    from_pixel_to_normalized = np.linalg.inv(from_normalized_to_pixel)

    H_in_normalized_units = from_pixel_to_normalized @ H_in_pixel_units @ from_normalized_to_pixel
    # because a homography matrix is only meaningful up to scalar multiples,
    # it is traditional to scale it to make the bottom left entry 1 as a kind of standard representative:
    H_in_pixel_units_standardized = H_in_normalized_units / H_in_normalized_units[2, 2]
    return H_in_pixel_units_standardized



def test_convert_pixel_units_homography_to_normalize_units_homography():
    points_in_image_a_pixel_units = np.array(
        [
            [292.0867614746094, 958.3085327148438],
            [912.1717529296875, 886.8214721679688],
            [1134.3668212890625, 515.5587158203125],
            [1764.6827392578125, 716.1578369140625],
        ]
    )

    points_in_image_b_pixel_units = np.array(
        [
            [10.94099047277401, 982.953231417786 ],
            [660.7944710962003, 893.0506025521464],
            [883.3228161272104, 509.5280975980616],
            [1516.0694386427815, 702.1927845848164],
        ]
    )


    H_in_pixel_units = np.array(
        [
            [1.0791474341656404, 0.014565876886892442, -318.12030238766613, ],
            [0.0017906935949507721, 1.054019909201253, -18.483120921094716, ],
            [2.987182286898967e-05, 6.22661556168849e-07, 1.0, ],
        ]
    )

    mapped_through_H_pixel_units = map_points_through_homography(points_in_image_a_pixel_units, H_in_pixel_units)
    print("First off, lets check the homography_in_pixel_units basically works: these should be the same:")
    print(mapped_through_H_pixel_units)
    print(points_in_image_b_pixel_units)
    assert (
        np.allclose(
            mapped_through_H_pixel_units,
            points_in_image_b_pixel_units
        )
    ), "WTF"

    points_in_image_a_normalized_units = (
        (
            points_in_image_a_pixel_units
            -
            np.array([1920/2, 1080/2])
        )
        / (1920/2)
    )

    points_in_image_b_normalized_units = (
        (
            points_in_image_b_pixel_units
            -
            np.array([1920/2, 1080/2])
        )
        / (1920/2)
    )

    H_in_normalized_units = convert_pixel_units_homography_to_normalized_units_homography(H_in_pixel_units)
    print(f"We claim that H_in_normalized_units =\n{H_in_normalized_units}")
    mapped_through_H_normalized_units = map_points_through_homography(
        points_in_image_a_normalized_units,
        homography_3x3=H_in_normalized_units
    )
    print("These better match, these should be the same:")
    print(mapped_through_H_normalized_units)
    print(points_in_image_b_normalized_units)
    assert (
        np.allclose(
            mapped_through_H_normalized_units,
            points_in_image_b_normalized_units
        )
    ), "WTF"


def make_rando_SO3_matrix(theta_degrees, phi_degrees):
    theta = theta_degrees / 180 * np.pi  # say 15 degrees rotation
    phi = phi_degrees / 180 * np.pi  # say 7 degrees rotation
    R = np.dot(
        np.array(
            [
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)]
            ]
        ),
        np.array(
            [
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ]
        )
    )
    return R


def world_coordinates_to_normalized_image_observation(xyz_world, camera_parameters):
    loc = camera_parameters["loc"]
    world_to_camera = camera_parameters["world_to_camera"]
    f = camera_parameters["f"]
    p_gicc = np.dot(world_to_camera, xyz_world - loc)

    x_gicc = p_gicc[0]
    y_gicc = p_gicc[1]
    z_gicc = p_gicc[2]

    x_over_z = x_gicc / z_gicc
    y_over_z = y_gicc / z_gicc
    u = f * x_over_z
    v = f * y_over_z
    return np.array([u, v])


def infer_camera_b_focal_length_and_world_to_camera(cp_a, points_in_image_a, points_in_image_b):
    f_a = cp_a.f
    wtc_a = cp_a.world_to_camera
    points_in_rays = np.zeros(shape=(points_in_image_a.shape[0], 3))
    print(points_in_image_a)
    points_in_rays[:, :2] = points_in_image_a
    points_in_rays[:, 2] = f_a
    # print(points_in_rays)
    norms = np.sqrt(np.sum(points_in_rays**2, axis=1))
    unit_rays_a = points_in_rays / norms[:, np.newaxis]
    # print(unit_rays_a)


    cos_of_angle_between_first_two_rays_a = np.dot(unit_rays_a[0], unit_rays_a[1])
    matrix_of_cos_angles_a = np.dot(unit_rays_a, unit_rays_a.T)
    
    # print("matrix_of_cos_angles_a:")
    # print(matrix_of_cos_angles_a)
    # print(f"The cos(angle) between the first two rays in a is {cos_of_angle_between_first_two_rays_a}")


    u0, v0 = points_in_image_b[0,:]
    u1, v1 = points_in_image_b[1, :]
    C = cos_of_angle_between_first_two_rays_a
    D = u0 * u1 + v0 * v1
    E = u0**2 + v0**2
    F = u1**2 + v1**2

    #(1-C^2) L^2 + [2D - C^2(E + F)] L + (D^2 - C^2 EF) = 0
    quadratic_a = (1-C**2) 
    quadratic_b = 2*D - C**2 * (E + F)
    quadratic_c = D**2 - C**2 * E * F
    L = (- quadratic_b + np.sqrt(quadratic_b**2 - 4 * quadratic_a * quadratic_c)) / (2 * quadratic_a)
    # print(f"L i.e. f_b^2 = {L}")
    f_b = np.sqrt(L)
    n0 = np.sqrt((u0**2 + v0**2 + f_b**2))
    n1 = np.sqrt((u1**2 + v1**2 + f_b**2))
    rays_intersect_image_plane_b = np.zeros(shape=(points_in_image_b.shape[0], 3))
    rays_intersect_image_plane_b[:, :2] = points_in_image_b
    rays_intersect_image_plane_b[:, 2] = f_b
    norms_b = np.sqrt(np.sum(rays_intersect_image_plane_b**2, axis=1))
    unit_rays_b = rays_intersect_image_plane_b / norms_b[:, np.newaxis]
    matrix_of_cos_angles_b = np.dot(unit_rays_b, unit_rays_b.T)
    # print("matrix_of_cos_angles_b:")
    
    print("HERE:")
    print(matrix_of_cos_angles_a - matrix_of_cos_angles_b)

    cos_of_angle_between_first_two_rays_b = np.dot(unit_rays_b[0], unit_rays_b[1]) 
    # print(f"The cos(angle) between the first two rays in b is {cos_of_angle_between_first_two_rays_b}")
    assert abs(cos_of_angle_between_first_two_rays_b - cos_of_angle_between_first_two_rays_a) < 1e-8

    # solve for R in least squares sense:  unit_rays_b * R = unit_rays_a
    R, residuals, _, _ = np.linalg.lstsq(unit_rays_b, unit_rays_a, rcond=None)
    R = np.linalg.inv(unit_rays_b.T @ unit_rays_b) @ (unit_rays_b.T @ unit_rays_a)
    # print(f"R =\n{R}")
    world_to_camera_b = np.dot(R, wtc_a)
    return f_b, world_to_camera_b



def infer_camera_b_focal_length_and_world_to_camera_from_homography(cp_a, H):
    """
    suppose 3x3 matrix H "expresses a homography" in the typical manner from image A to image B,
    i.e. is such that for all observations (u, v) in image A
    (u', v') is the corresponding spot in image B where
    
    alpha * [u', v', 1]^T = H * [u, v, 1]^T,
    
    or, said more explicitly, calculate c, and d, e via
    
    [c, d, e]^T = H * [u, v, 1]^T.
    
    and then
        u' = c / e
    and
        v' = d / e.
    Then, given that both cameras have the same location,
    this tells you what camera B's focal length and world_to_camera matrix are.
    """
    points_in_image_a = np.random.randn(1000, 2)
    #     [
    #         [1, 0],
    #         [0, 1],
    #         [1, 1]
    #     ]
    # ) # we pick three points in image_a [1, 0] and [0, 1]

    points_in_image_b = map_points_through_homography(inhomo_points_as_rows=points_in_image_a, homography_3x3=H)
    # print(points_in_image_b)
    focal_length_b, world_to_camera_b = infer_camera_b_focal_length_and_world_to_camera(
        cp_a=cp_a,
        points_in_image_a=points_in_image_a,
        points_in_image_b=points_in_image_b
    )
    return focal_length_b, world_to_camera_b




def test_it_works():
    # make a camera_parameters_a:
    f_a = 3 # 1 + np.random.rand() * 3
    loc_a = np.zeros(shape=(3,))
    wtc_a = make_rando_SO3_matrix(7, -3)
    cp_a = dict(
        f=1,
        world_to_camera=wtc_a,
        loc=loc_a
    )

    # make another camera_parameters_b, located at the same location (camera is on a tripod), rotated differently, different focal length as well
    wtc_b = make_rando_SO3_matrix(14, 11)
    f_b = 2 #1 + np.random.rand() * 3
    loc_b = loc_a  # camera is on a tripod, this is important
    cp_b = dict(
        f=f_b,
        world_to_camera=wtc_b,
        loc=loc_b
    )

    # pick some points in the world coordinate system
    xyz_world_points = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.2, 0.3, 1],
            [0.1, 0.2, 2],
            [0.2, -0.1, 1],
            [0.3, 0.1, 1],
            [0.5, 0.9, 5], 

        ]
    )
    num_correspondences = xyz_world_points.shape[0]
    points_in_image_a = np.zeros(shape=(num_correspondences, 2))
    points_in_image_b = np.zeros(shape=(num_correspondences, 2))

    for point_index, xyz_world in enumerate(xyz_world_points):
        uv_a = world_coordinates_to_normalized_image_observation(xyz_world=xyz_world, camera_parameters=cp_a)
        print(uv_a)

        points_in_image_a[point_index, :] = uv_a

        uv_b = world_coordinates_to_normalized_image_observation(xyz_world=xyz_world, camera_parameters=cp_b)
        print(uv_b)
        points_in_image_b[point_index, :] = uv_b
    
    print(points_in_image_a)
    print(points_in_image_b)

    H = solve_for_homography_based_on_4_point_correspondences(
        points_in_image_a=points_in_image_a,
        points_in_image_b=points_in_image_b
    )

    print(f"Notice that homography H =\n{H}\nmaps the points_in_image_a to the points_in_image_b:")

    print(f"actual points_in_image_b =\n{points_in_image_b}")
    
    mapped_through_H = map_points_through_homography(points_in_image_a, H)
    print(f"The homography maps the camera_a observations to:\n{mapped_through_H}")


     

    # recovered_focal_length_camera_b, recovered_world_to_camera_b = infer_camera_b_focal_length_and_world_to_camera(
    #     cp_a=cp_a,
    #     points_in_image_a=points_in_image_a,
    #     points_in_image_b=points_in_image_b
    # )

    # print(f"recovered_world_to_camera_b =\n{recovered_world_to_camera_b}")
    # print(f"world_to_camera_b actually was:\n{wtc_b}")
    

    recovered_focal_length_camera_b, recovered_world_to_camera_b = infer_camera_b_focal_length_and_world_to_camera_from_homography(
        cp_a=cp_a,
        H=H        
    )

    print(f"recovered_world_to_camera_b is:\n{recovered_world_to_camera_b}")
    print(f"recovered_focal_length_camera_b is:\n{recovered_focal_length_camera_b}")
    print(f"world_to_camera_b actually was:\n{wtc_b}")
    print(f"world_to_camera_b actually was:\n{f_b}")


def test_actual_data():
    """
    We have some video frames taken from a tripod where we know the camera parameters.
    We test that camera_parameters_a "+" homography_from_a_to_b equals camera_parameters_b.
    """
    cp_a = {
        "rod": [1.7110034230945796, 0.1632188640245258, -0.1587924370303849],
        "loc": [-2.6000000007441457, -78.50000000035067, 17.499999999956007],
        "f": 2.845725717709582
    }
    cp_a["rod"] = np.array(cp_a["rod"])
    cp_a["world_to_camera"] = SO3_from_rodrigues(cp_a["rod"])
    

    cp_b_should_be= {
        "rod": [1.7187967325416136, 0.08698888051011214, -0.09071260296132387],
        "loc": [-2.6000000822855434, -78.5000000151574, 17.667414226035582],
        "f": 2.8505431509170305,
    }
    cp_b_should_be["rod"] = np.array(cp_b_should_be["rod"])
    cp_b_should_be["world_to_camera"] = SO3_from_rodrigues(cp_b_should_be["rod"])
    world_to_camera_b_should_be = cp_b_should_be["world_to_camera"]

    # using SIFT in opencv, we get this:
    H_in_pixel_units = np.array(
        [
            [1.0791474341656404, 0.014565876886892442, -318.12030238766613, ],
            [0.0017906935949507721, 1.054019909201253, -18.483120921094716, ],
            [2.987182286898967e-05, 6.22661556168849e-07, 1.0, ],
        ]
    )

    H_in_pixel_units = np.array(
        [
            [1.2158355370214498, 0.0342410961412755, -786.3072179876935, ],
            [0.003765353545696503, 1.1674700591375742, -56.336987961436215, ],
            [6.957381084186135e-05, 4.5531985188263074e-07, 1.0, ],
        ]
    )

    H_in_pixel_units = np.array(
        [
            [1.4225319952852484, 0.07020579647779379, -1574.7191674987728, ],
            [0.002252213310619955, 1.3697601775502675, -105.80662884787013, ],
            [0.00012798110681503926, 2.1670210056148134e-06, 1.0, ],
        ]
    )

    H_in_normalized_units = convert_pixel_units_homography_to_normalized_units_homography(H_in_pixel_units)
    predicted_f_b, predicted_world_to_camera_b = infer_camera_b_focal_length_and_world_to_camera_from_homography(cp_a, H_in_normalized_units)
    print(f"We predict f_b={predicted_f_b}")
    rod_b = rodrigues_from_SO3(predicted_world_to_camera_b)
    print(f"we predict world_to_camera_b=\n{predicted_world_to_camera_b}")
    print(f"i.e. we predict rodrigues vector {rod_b}")
    print(f"It should have been:\n{world_to_camera_b_should_be}")
    print("The difference is:")
    print(predicted_world_to_camera_b - world_to_camera_b_should_be)
    print(f"Whereas camera a's world to camera was:")
    print(cp_a["world_to_camera"] - world_to_camera_b_should_be)




if __name__ == "__main__":
   # test_convert_pixel_units_homography_to_normalize_units_homography()
   test_it_works()
   # test_actual_data()