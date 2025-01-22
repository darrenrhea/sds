import numpy as np
import cv2
from rodrigues_utils import SO3_from_rodrigues, rodrigues_from_SO3


# from camera_solve_contexts import get_geometry
# from CameraParameters import CameraParameters
from recover_SO3_from_shadow import recover_SO3_from_its_shadow, recover_SO3_from_its_shadow_aa

__all__ = [
    "map_points_through_homography",
    "evaluate_homography",
    "report_homography_error",
    "find_best_homography_explaining_2d_correspondences",
    "map_points_through_homography",
]

def evaluate_homography(
    H,
    src_points_in_pixel_units,
    dst_points_in_pixel_units
):
    assert isinstance(H, np.ndarray)
    assert H.shape == (3, 3)
    assert isinstance(src_points_in_pixel_units, np.ndarray)
    assert isinstance(dst_points_in_pixel_units, np.ndarray)
    assert src_points_in_pixel_units.shape == dst_points_in_pixel_units.shape
    assert src_points_in_pixel_units.shape[1] == 2
    print(f"Evaluating the homography {H=}")

    num_points = len(src_points_in_pixel_units)

    mapped = map_points_through_homography(
        points=src_points_in_pixel_units,
        homography=H
    )
    errors = np.sqrt(np.sum((mapped - dst_points_in_pixel_units)**2, axis=1))
    assert errors.shape == (num_points, )
    inlier_indices = np.argwhere(errors < 1)[:, 0]
    assert inlier_indices.ndim == 1
    print(f"inlier_indices: {inlier_indices}")
    num_inliers = inlier_indices.shape[0]
    if num_inliers < 3:
        return False, None, None
    print(f"num_inliers: {num_inliers}")
    print(f"percentage inliers: {num_inliers / num_points}")
    print(f"average error over inliers: {np.mean(errors[inlier_indices])}")
    total_error = np.sum(errors)
    print(f"total_error: {total_error}")
    return True, {
        "num_inliers": num_inliers,
        "percentage_inliers": num_inliers / num_points,
        "average_error_over_inliers": np.mean(errors[inlier_indices]),
        "total_error": total_error
    }, inlier_indices
    
    

def convert_normalized_homography_back_to_pixel_units(H, width, height):
    """
    Given standardized version of H, i.e. one that works on
    normalized coordinates in  [-1, 1] x [-9/16, 9/16]
    Return a version of H that works on pixel coordinates in [0, 1280] x [0, 720]
    """
    assert H.shape == (3, 3)
    ratio = height / width
   
    H1 = np.array([
        [1, 0, 1],
        [0, 1, ratio],
        [0, 0, 2/width],
    ])
    H2 = np.linalg.inv(H1)

    ans  = H1 @ H @ H2
   
    return ans

def convert_homography_H_to_normalized_coordinates(H, width, height):
    """
    Given an H that maps grossly [0, 1280] x [0, 720] to [0, 1280] x [0, 720]
    Return a standardized version of H, such that
    instead of mapping [0, 1280] x [0, 720] to [0, 1280] x [0, 720]
    maps [-1, 1] x [-9/16, 9/16]
    """
    ratio = height /width
    # x_out = (x_in + 1) * width / 2
    # y_out = (y_in + ratio) * width / 2
    H1 = np.array([
        [1, 0, 1],
        [0, 1, ratio],
        [0, 0, 2/width],
    ])
    H2 = np.linalg.inv(H1)

    ans  = H2 @ H @ H1
    # points = np.array([
    # [-1, -9/16],
    # [-1, 9/16],
    # [1, -9/16],
    # [1, 9/16],
    # ])       
    
    # quad = map_points_through_homography(points, homography=ans)
    # print(quad)
    return ans

    

def angle_of_view_from_measurements():
    """
    exiftool seems to be saying "Field of View" when what they mean "horizontal angle of view".
    So-called "35 mm" actually means a 24mm x 36mm sensor,
    which has a a 2:3 aspect ratio, and whose diagonal is np.sqrt(24**2 + 36**2) = 43.2666 millimeters.

    It "understands" the iPhone X's x 1 camera as being "equivalent" to a "35mm" camera at focal length 29mm,
    where by equivalent we mean having approximately the same diagonal-angle-of-view.
    Against sites like
    https://www.nikonians.org/reviews/fov-tables
    you can confirm that these equations are how one converts from a certain focal length "35mm" camera
    to its horizontal, vertical, and diagonal angles of view.
    But ultimately this is not going to be very valid, because the aspect ratio of a "35mm" camera
    is very different from an iPhones, so the three angles cannot all match, only at most one,
    not to mention that they only do integer millimeter focal lengths, whereas it would need to be fractional
    to match even the diagonal angle-of-view.
 
    So the (diagonal) angle of view of 29 mm focal length for "35 mm" sensor is
    180 / np.pi * 2 * np.arctan(43.2666  / 2 / 29) = 73.444 degrees diagonal
    180 / np.pi * 2 * np.arctan(36  / 2 / 29) = 63.654 degrees wide (this explains the 63.7 degrees in exiftool)
    180 / np.pi * 2 * np.arctan(24  / 2 / 29) = 44.958 degrees tall
    
    unlike the 4:3 aspect ratio of the iphone sensor which is 4.80mm x 3.60mm with focal length 4.0 mm.
    leading us to 
    horizontal_angle_of_view = 180 / np.pi * 2 * np.arctan(4.8  / 2 / 4) = 61.927 degrees wide
    vertical_angle_of_view = 180 / np.pi * 2 * np.arctan(3.6  / 2 / 4) = 48.455 degrees tall
    sensor_diagonal = np.sqrt(4.8**2 + 3.6**2) = 6 mm (half the sensor is 1.2 mm times the 3-4-5 triangle)
    thus
    diagonal_angle_of_view = 180 / np.pi * 2 * np.arctan(6 / 2 / 4) = 73.739 degrees
    This has good correspondence with experimental measurements of measuring tapes.
    Also the focal length in pixels can be calculated as either
    4032 * 4 / 4.8 = 3360 pixels focal length
    or
    3024 * 4 / 3.6 = 3360 pixels focal length.
    Note that when filming video which is 16:9 aspect ratio, it does-not / cannot use the whole sensor.
    """
    measured_distance_from_the_camera = 27.908
    width_visible = 33.5325 # should be 27.9 * 4.8 / 4 = 33.48
    height_visible = 25.125  # should be 27.9 * 3.6 / 4 = 25.11
    diagonal_visible = np.sqrt(width_visible**2 + height_visible**2)
    print(f"diagonal_visible = {diagonal_visible}")
    image_width_in_pixels = 4032
    image_height_in_pixels = 3024
    2261.49
    horizontal_angle_of_view_in_degrees = \
        180/np.pi * 2 * np.arctan(width_visible/2/measured_distance_from_the_camera)
    print(
        f"At a distance of {measured_distance_from_the_camera} from the camera, "
        f"we can see {width_visible} wide, so that's an angle of "
        f"{horizontal_angle_of_view_in_degrees}"
    )
    vertical_angle_of_view_in_degrees = \
        180/np.pi * 2 * np.arctan(height_visible/2/measured_distance_from_the_camera)
    print(
        f"At a distance of {measured_distance_from_the_camera} from the camera, "
        f"we can see {height_visible} high, so that's an angle of "
        f"{vertical_angle_of_view_in_degrees}"
    )

    diagonal_angle_of_view_in_degrees = \
        180/np.pi * 2 * np.arctan(
            np.sqrt((height_visible/2)**2 + (width_visible/2)**2) /measured_distance_from_the_camera
        )
    print(
        f"measuring the diagonal, we get {diagonal_angle_of_view_in_degrees} diagonal_angle_of_view_in_degrees"
    )
    print(f"At a distance of 1 we can see {width_visible/measured_distance_from_the_camera}, so")
    print(f"so probably for a focal length of 1 you should describe the x coords as varying from +-{width_visible/measured_distance_from_the_camera/2}")




def angle_between_vectors_a_and_b_in_degrees(a, b):
    # assert np.abs(np.linalg.norm(a) - 1.0) < 1e-12, f"{np.linalg.norm(a)}"
    # assert np.abs(np.linalg.norm(b) - 1.0) < 1e-12, f"{np.linalg.norm(b)}"
    return 180 / np.pi * np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


def x1_y1_x2_y2_line_to_l(x1_y1_x2_y2):
    """
    Annotators usually describe a line on an image via
    two distinct points (x1,y1) and (x2,y2) within the line.
    Instead, we want to describe the line as the locus of all
    points (x,y) such that (x,y,1) dot products with L to give zero.
    """
    x1, y1, x2, y2 = x1_y1_x2_y2
    a = np.array([x1, y1, 1], dtype=np.double)
    b = np.array([x2, y2, 1], dtype=np.double)
    L = np.cross(a=a, b=b)  # cross product
    L /= L[2]  # L does not matter up to scale, so make its last coordinate positive 1
    L /= np.linalg.norm(L)
    assert L[2] > 0
    assert np.abs(np.linalg.norm(L) - 1) < 1e-12
    return L


def map_lines_through_homography(lines, H):
    """
    Given a bunch of projective lines as rows in an I x 3 matrix,
    map them through the homography.
    
    if a line is the locus of all homogenous points p = [x, y, 1] s.t.
    L^T p = 0, then L^T H^{-1} H p = 0, i..e (H^{-1}^T * L)^T H p = 0.
    """
    ans = np.dot(lines, np.linalg.inv(H))  # I x 3
    scalars = np.sum(ans ** 2, axis=1) ** 0.5
    # scalars = ans[:, 2][:, np.newaxis]
    ans /= scalars
    return ans


def map_points_through_homography(
    points: np.ndarray,  # a numpy array of 2D points (either every row is a point, so it is a N x 2 array, or every column is a point, so it is a 2 x N array)
    homography: np.ndarray,  # the 3 x 3 homography matrix you want to use as a map/function from R^2 to R^2
    a_point_is_a_row: bool = True  # whether points is a N x 2 array or a 2 x N array
) -> np.ndarray:
    """
    The use of cv2.perspectiveTransform is slightly strange
    in that you need an extra trivial dimension in the input.
    """
    assert points.ndim == 2
    if a_point_is_a_row:
        assert points.shape[1] == 2, "num_columns must be 2 if a_point_is_a_row"
    else:
        assert points.shape[0] == 2, "num_rows must be 2 if a_point_is_a_column"
    
    assert points.dtype == np.float32 or points.dtype == np.float64
    if points.dtype == np.float32:
        precision = "float32"
    else:
        precision = "float64"
    
    assert homography.shape == (3, 3)
    # assert homography.dtype == np.float32
    if a_point_is_a_row:
        reshaped_points = points.reshape(-1, 1, 2)
    else:
        reshaped_points = points.T.reshape(1, -1, 2)
    
    if a_point_is_a_row:
        mapped_points = cv2.perspectiveTransform(
            src=reshaped_points,
            m=homography
        ).reshape(-1, 2)
    else:
        mapped_points = cv2.perspectiveTransform(
            src=reshaped_points,
            m=homography
        ).T.reshape(2, -1)

    if precision == "float32":
        assert mapped_points.dtype == np.float32
    else:
        assert mapped_points.dtype == np.float64
    
    if a_point_is_a_row:
        assert mapped_points.shape[1] == 2
    else:
        assert mapped_points.shape[0] == 2
    return mapped_points


def map_points_through_homography_variant(points, homography_3x3):
    """
    Given N (inhomogeneous) points as the rows of an N x 2 numpy array of doubles,
    adds a column of 1s to homogenize them;
    maps them through the homography x' = H x,
    de-homogenizes them back to an N x 2 numpy array of doubles.
    """
    assert points.shape[1] == 2
    homogeneous_points = np.ones(
        shape=(points.shape[0], 3), dtype=np.double
    )
    homogeneous_points[:, :2] = points
    # print(
    #     f"If you homogenize the points:\n{points},\nyou get:\n{homogeneous_points}\n"
    # )
    homogeneous_answer = np.dot(homography_3x3, homogeneous_points.T).T
    # print(f"This maps through homography H =\n{H}\nto give:\n{homogeneous_answer}")
    scalars = homogeneous_answer[:, 2]
    homogeneous_answer /= scalars[:, np.newaxis]
    # print(f"Which is the same up to scaling as:\n{homogeneous_answer}")
    answer = homogeneous_answer[:, :2]
    return answer


def report_homography_error(
    matched_source_points: np.ndarray,
    homography_3x3: np.ndarray,
    matched_destination_points: np.ndarray
):
    """
    Given a matching between some 2D points in the source image and some 2D points in the destination image,
    this evaluates how close the homography comes to mapping the source points onto the destination points.
    matched_source_points: a numpy array of 2D points in the source image (every row is a point)
    homography_3x3: The 3x3 matrix defining the homography
    matched_destination_points: a numpy array of 2D points in the destination image (every row is a point)
    """

    assert matched_source_points.shape[1] == 2
    assert matched_destination_points.shape[1] == 2
    assert matched_source_points.shape[0] == matched_destination_points.shape[0]
    predictions = map_points_through_homography(
        points=matched_source_points, 
        homography=homography_3x3
    )
    residuals = predictions - matched_destination_points
    l2 = np.sum(np.abs(residuals**2), axis=1)
    I = np.argsort(l2)
    l2 =l2[I]
    for prediction, destination_point in zip(predictions, matched_destination_points):
        pass
        # print(prediction)
        # print(destination_point)
    return l2



def solve_for_homography_based_on_4_point_correspondences(
    points_in_image_a,
    points_in_image_b
):
    """
    s * [u' v' 1]^T and  [h0 h1 h2; h3 h4 h5; h6 h7 h8][u v 1] are linearly dependent
    for all four pairs (u, v, u' v').

    The cross-product of [u*h0 + v*h1 + h2, u*h3 + v*h4 + h5, u*h6 + v*h7 + h8] with [u' v' 1]
    must be zero.
    (u*h0 + v*h1 + h2)*1 - (u*h6 + v*h7 + h8) * u' = 0
    (u*h3 + v*h4 + h5)*1 - (u*h6 + v*h7 + h8) * v' = 0

    [u v 1 0 0 0 -u*u' -v*u' -u']
    [0 0 0 u v 1 -u*v' -v*v' -v']
    """
    M = np.zeros(shape=(9, 9))
    for k in range(4):
        u = points_in_image_a[k, 0]
        v = points_in_image_a[k, 1]
        u_prime = points_in_image_b[k, 0]
        v_prime = points_in_image_b[k, 1]
        M[2 * k, :] = np.array(
            [u, v, 1, 0, 0, 0, -u * u_prime, -v * u_prime, -u_prime], dtype=np.double
        )
        M[2 * k + 1, :] = np.array(
            [0, 0, 0, u, v, 1, -u * v_prime, -v * v_prime, -v_prime], dtype=np.double
        )
    M[8, :] = np.random.randn(9)
    one_in_last_spot = np.zeros(shape=(9,))
    one_in_last_spot[8] = 100
    h = np.linalg.solve(M, one_in_last_spot)

    H = h.reshape((3, 3))
    H /= H[2, 2]

    answer = map_points_through_homography(
        points=points_in_image_a,
        homography=H
    )
  
    return H


# def somehow_get_homography_from_video_frame_to_world_floor(
#     frame_index,
#     ransacReprojThreshold
# ):
#     """
#     We need to select a finite set of photos of the scene to be the keyframes.
#     For each "keyframe", we need to know the homography from that keyframe
#     to the world-coordinates-floor-space xy-plane-in-feet.
#     What that procedure is exactly, we don't know yet, but this is one way,
#     based upon having known-location landmarks annotated in the keyframe.
#     """
#     geometry = get_geometry("ncaa_kansas")  # we need to know for each named landmark where it is in world coordinate system


#     annotations_json_path = Path(
#         f"~/awecom/data/clips/swinney1/landmark_locations_for_homographies/swinney1_{frame_index:06d}_landmark_locations_for_homographies.json"
#     ).expanduser()

#     annotations_json = bj.load(annotations_json_path)
    
#     pixel_points = annotations_json["pixel_points"]
    
#     landmark_name_to_photo_location = {
#         landmark_name: [dct["j"], dct["i"]]
#         for landmark_name, dct in pixel_points.items()
#     }
    
#     landmark_names = sorted([key for key in pixel_points])

#     xy_points_in_the_photo = np.array(
#         [
#             landmark_name_to_photo_location[landmark_name]
#             for landmark_name in landmark_names
#         ]
#     )

#     xy_points_in_the_world_floor = np.array(
#         [
#             [geometry["points"][landmark_name][0], geometry["points"][landmark_name][1]]
#             for landmark_name in landmark_names
#         ] 
#     )

#     ans = find_homography_from_2d_correspondences(  # call opencv
#         source_points=xy_points_in_the_photo,
#         destination_points=xy_points_in_the_world_floor,
#         ransacReprojThreshold=ransacReprojThreshold # what units is this? feet? pixels???
#     )

#     homography_from_photo_to_world_floor = ans["homography_3x3"]

#     report_homography_error(
#         matched_source_points=xy_points_in_the_photo,
#         homography_3x3=homography_from_photo_to_world_floor,
#         matched_destination_points=xy_points_in_the_world_floor
#     )

#     return homography_from_photo_to_world_floor



def get_camera_parameters_that_cause_this_homography_from_pixel_coordinates_to_world_floor(
    homography_from_pixel_coordinates_to_world_floor
):
    pass


def get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters(
    camera_parameters
):
    """
    If a camera has no distortion, the way it maps the world floor into the photo
    should be a homography, as should its inverse be a homography.
    This returns a camera parameters that has two focal lengths fx and fy
    The distortionless but two-focal-lengthed camera perspective projection process goes
    like this:
    Take a world coordinates point , say [x_w y_w z_w]^T that you want to perspective project.
    Append a 1 to it to make the 4x1 column vector [x_w y_w z_w 1]^T.
    Say the camera's in the world coordinate system is c_w = [xc_w, yc_w, zc_w]^T.
    The change of coordinate system into the camera coordinate system
    can be accomplished as follows: there is an SO3 3x3 matrix R = world_to_camera
    together with a 3x1 T := -R * c_w that causes the change of coordinates:
    [x_c y_c z_c]^T = R * ([x_w y_w z_w]^T - [cx_w cy_w cz_w]^T)
    p_c = R * (p_w - c_w)
    p_c =R * p_w + T 
    Make a 4x4 matrix B out of blocks: 
    B = [R_3x3 T_3x1]
        [0_1x3 1_1x1]
    
    the purpose of B is the change of coordinates, i.e. notice that
    P_c = [x_c y_c z_c 1]^T = B * [x_w y_w z_w 1]^T = B * P_w
    Let K be a 3x4 via
    K = [fx   0  0    0]
        [ 0  fy  0    0]
        [ 0      1    0]
    z_c * [un vn 1]^T = K * B *  P_w
    where un and vn are normalized image coordinates.
    To get to unnormalized, we need to multiply by this 3x3

    J = [1920/2      0  1920/2]
        [     0 1920/2  1080/2]
        [     0      0       1]
    so that 
    J * z_c * [un vn 1]^T = J * K * B * P_w =
    z_c * [1920/2 * un + 1920/2, 1920/2 * vn + 1920/2, 1]^T
    = z_c * [u_pixel, v_pixel, 1]^T
    """
    cp = camera_parameters
    x_w = np.random.randn()
    y_w = np.random.randn()
    z_w = 0
    p_w = np.array([[x_w], [y_w], [z_w]])
    P_w = np.array([[x_w], [y_w], [z_w], [1]])
   
    J = np.array(
        [
            [1920/2,      0, 1920/2],
            [     0, 1920/2, 1080/2],
            [     0,      0,      1],
        ]
    )
    # note J is invertible
    fx = cp.fx
    fy = cp.fy

    K = np.array(
        [
            [fx,   0,  0,    0],
            [ 0,  fy,  0,    0],
            [ 0,   0,  1,    0],
        ]
    )
   
    R = SO3_from_rodrigues(cp.rod)
    c_w = cp.loc
    T = - R @ c_w
    B = np.zeros(shape=(4, 4))
    B[:3, :3] = R
    B[:3,  3] = T
    B[3, 3] = 1

    """
    K B = 
                               [R00 R01 R02 T0]
            [fx,   0,  0, 0]   [R10 R11 R12 T1]  =  [fxR00 fxR01 fxR02 fxT0]
            [ 0,  fy,  0, 0] * [R20 R21 R22 T2]     [fyR10 fyR11 fyR12 fyT1]
            [ 0,   0,  1, 0]   [  0   0   0  1]     [  R20   R21   R22   T2]
    """

    # There is an I_4x3 such that
    # [x_w, y_w, 1]^T = I_4x3 * [x_w, y_w, 0, 1]^T
    I = np.zeros(shape=(4, 3))
    I[0, 0] = 1
    I[1, 1] = 1
    I[3, 2] = 1
    """
    I_4x3 = [1 0 0]
            [0 1 0]
            [0 0 0]
            [0 0 1]

    [T0] = [ 
    [T1] = [
    [T2] = [

    K B I = [fxR00 fxR01 fxT0]
            [fyR10 fyR11 fyT1]
            [  R20   R21   T2]
    """


    """
    [x_w, y_w, 0, 1]^T = [1 0 0] * [x_w, y_w, 1]^T
                         [0 1 0]
                         [0 0 0]
                         [0 0 1]
    
    z_c * [u_pixel, v_pixel, 1]^T = J * K * B * I * [x_w, y_w, 1]^T
    """
    world_floor_coordinates_to_photo_pixel_homography = J @ K @ B @ I
    return world_floor_coordinates_to_photo_pixel_homography
  

def get_homography_from_pixel_coordinates_to_world_floor_from_distortionless_camera_parameters(
    camera_parameters
):
    world_floor_coordinates_to_photo_pixel_homography = get_homography_from_world_floor_to_photo_pixel_from_distortionless_camera_parameters(
        camera_parameters
    )

    photo_pixel_to_world_floor_coordinates_homography = np.linalg.inv(world_floor_coordinates_to_photo_pixel_homography)
    return photo_pixel_to_world_floor_coordinates_homography


def get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel(
    H
):
    """
    The trouble with this is that is has to have two focal lengths fx and fy
    since a homography has 8 degrees of freedom whereas the camera_parameters only has 7 degrees of freedom.
    If a camera has no distortion, the way it maps the world floor into the photo
    should be a homography, as should its inverse be a homography.
    This returns a camera parameters that has two focal lengths fx and fy
    The distortionless but two-focal-lengthed camera perspective projection process goes
    like this:
    Take a world coordinates point , say [x_w y_w z_w]^T that you want to perspective project.
    Append a 1 to it to make the 4x1 column vector [x_w y_w z_w 1]^T.
    Say the camera's in the world coordinate system is c_w = [xc_w, yc_w, zc_w]^T.
    The change of coordinate system into the camera coordinate system
    can be accomplished as follows: there is an SO3 3x3 matrix R = world_to_camera
    together with a 3x1 T := -R * c_w that causes the change of coordinates:
    [x_c y_c z_c]^T = R * ([x_w y_w z_w]^T - [cx_w cy_w cz_w]^T)
    p_c = R * (p_w - c_w)
    p_c =R * p_w + T 
    Make a 4x4 matrix B out of blocks: 
    B = [R_3x3 T_3x1]
        [0_1x3 1_1x1]
    
    the purpose of B is the change of coordinates, i.e. notice that
    P_c = [x_c y_c z_c 1]^T = B * [x_w y_w z_w 1]^T = B * P_w
    Let K be a 3x4 via
    K = [fx   0  0    0]
        [ 0  fy  0    0]
        [ 0      1    0]
    z_c * [un vn 1]^T = K * B *  P_w
    where un and vn are normalized image coordinates.
    To get to unnormalized, we need to multiply by this 3x3

    J = [1920/2      0  1920/2]
        [     0 1920/2  1080/2]
        [     0      0       1]
    so that 
    J * z_c * [un vn 1]^T = J * K * B * P_w =
    z_c * [1920/2 * un + 1920/2, 1920/2 * vn + 1920/2, 1]^T
    = z_c * [u_pixel, v_pixel, 1]^T

    K B = 
                               [R00 R01 R02 T0]
            [fx,   0,  0, 0]   [R10 R11 R12 T1]  =  [fxR00 fxR01 fxR02 fxT0]
            [ 0,  fy,  0, 0] * [R20 R21 R22 T2]     [fyR10 fyR11 fyR12 fyT1]
            [ 0,   0,  1, 0]   [  0   0   0  1]     [  R20   R21   R22   T2]
   
    I_4x3 = [1 0 0]
            [0 1 0]
            [0 0 0]
            [0 0 1]


    K B I = [fxR00 fxR01 fxT0]
            [fyR10 fyR11 fyT1]
            [  R20   R21   T2]
    
    So the homography 3x3 equals this.

    s K B I = [s*fxR00 s*fxR01 s*fxT0]
              [s*fyR10 s*fyR11 s*fyT1]
              [  s*R20   s*R21   s*T2]
    We solve this for R, s, fx, fy, T0, T1, T2
    """
    world_floor_coordinates_to_photo_pixel_homography = H
    # standardize it:
    world_floor_coordinates_to_photo_pixel_homography /= H[2,2]
    
    J = np.array(
        [
            [1920/2,      0, 1920/2],
            [     0, 1920/2, 1080/2],
            [     0,      0,      1],
        ]
    )
    # note J is invertible

    sKBI = np.linalg.inv(J) @ world_floor_coordinates_to_photo_pixel_homography
    print(f"sKBI={sKBI}")
    conditioning_factor = 1000
    the_shadow_of_R = sKBI[:, :2] * conditioning_factor
    success, recovered_R, a, b, c = recover_SO3_from_its_shadow(the_shadow_of_R)
    
    if not success:
        return dict(
            success=False,
            rod=None,
            loc=None,
            fx=None,
            fy=None,
            world_to_camera=None
        )
    print(f"the recovered_R=\n{recovered_R}")
    s = sKBI[2, 0] / recovered_R[2, 0]
    also_s = sKBI[2, 1] / recovered_R[2, 1]
    print(f"s={s}")
    print(f"also_s={also_s}")
    fx = 1 / (a * s * conditioning_factor)
    fy = 1 / (b * s * conditioning_factor)
    print(f"fx={fx}, fy={fy}")
    recovered_T = np.zeros(3)
    recovered_T[0] = sKBI[0, 2] / (fx * s) 
    recovered_T[1] = sKBI[1, 2] / (fy * s)
    recovered_T[2] = sKBI[2, 2] / s
    recovered_loc = - recovered_R.T @ recovered_T
    # print(f"recovered loc = {recovered_loc}")
    recovered_rod = rodrigues_from_SO3(recovered_R)
    # print(f"recovered_rod = {recovered_rod}")
    return dict(
        success=True,
        rod=recovered_rod,
        loc=recovered_loc,
        fx=fx,
        fy=fy,
        world_to_camera=recovered_R
    )



def get_distortionless_camera_parameters_from_homography_from_world_floor_to_photo_pixel_only_one_focal_length(
    H
):
    """
    Trying to have only one focal length, which means it will not reproduce the homography exactly
    unless the homography had both focal lengths the same already.

    The trouble with this is that is has to have two focal lengths fx and fy
    since a homography has 8 degrees of freedom whereas the camera_parameters only has 7 degrees of freedom.
    If a camera has no distortion, the way it maps the world floor into the photo
    should be a homography, as should its inverse be a homography.
    This returns a camera parameters that has two focal lengths fx and fy
    The distortionless but two-focal-lengthed camera perspective projection process goes
    like this:
    Take a world coordinates point , say [x_w y_w z_w]^T that you want to perspective project.
    Append a 1 to it to make the 4x1 column vector [x_w y_w z_w 1]^T.
    Say the camera's in the world coordinate system is c_w = [xc_w, yc_w, zc_w]^T.
    The change of coordinate system into the camera coordinate system
    can be accomplished as follows: there is an SO3 3x3 matrix R = world_to_camera
    together with a 3x1 T := -R * c_w that causes the change of coordinates:
    [x_c y_c z_c]^T = R * ([x_w y_w z_w]^T - [cx_w cy_w cz_w]^T)
    p_c = R * (p_w - c_w)
    p_c =R * p_w + T 
    Make a 4x4 matrix B out of blocks: 
    B = [R_3x3 T_3x1]
        [0_1x3 1_1x1]
    
    the purpose of B is the change of coordinates, i.e. notice that
    P_c = [x_c y_c z_c 1]^T = B * [x_w y_w z_w 1]^T = B * P_w
    Let K be a 3x4 via
    K = [fx   0  0    0]
        [ 0  fy  0    0]
        [ 0      1    0]
    z_c * [un vn 1]^T = K * B *  P_w
    where un and vn are normalized image coordinates.
    To get to unnormalized, we need to multiply by this 3x3

    J = [1920/2      0  1920/2]
        [     0 1920/2  1080/2]
        [     0      0       1]
    so that 
    J * z_c * [un vn 1]^T = J * K * B * P_w =
    z_c * [1920/2 * un + 1920/2, 1920/2 * vn + 1920/2, 1]^T
    = z_c * [u_pixel, v_pixel, 1]^T

    K B = 
                               [R00 R01 R02 T0]
            [fx,   0,  0, 0]   [R10 R11 R12 T1]  =  [fxR00 fxR01 fxR02 fxT0]
            [ 0,  fy,  0, 0] * [R20 R21 R22 T2]     [fyR10 fyR11 fyR12 fyT1]
            [ 0,   0,  1, 0]   [  0   0   0  1]     [  R20   R21   R22   T2]
   
    I_4x3 = [1 0 0]
            [0 1 0]
            [0 0 0]
            [0 0 1]


    K B I = [fxR00 fxR01 fxT0]
            [fyR10 fyR11 fyT1]
            [  R20   R21   T2]
    
    So the homography 3x3 equals this.

    s K B I = [s*fxR00 s*fxR01 s*fxT0]
              [s*fyR10 s*fyR11 s*fyT1]
              [  s*R20   s*R21   s*T2]
    We solve this for R, s, fx, fy, T0, T1, T2
    """
    world_floor_coordinates_to_photo_pixel_homography = H
    # standardize it:
    world_floor_coordinates_to_photo_pixel_homography /= H[2,2]
    
    J = np.array(
        [
            [1920/2,      0, 1920/2],
            [     0, 1920/2, 1080/2],
            [     0,      0,      1],
        ]
    )
    # note J is invertible

    sKBI = np.linalg.inv(J) @ world_floor_coordinates_to_photo_pixel_homography
    # print(f"sKBI={sKBI}")
    conditioning_factor = 1
    the_shadow_of_R = sKBI[:, :2] * conditioning_factor
    success, recovered_R, a, b, c = recover_SO3_from_its_shadow_aa(the_shadow_of_R)
    
    if not success:
        return dict(
            success=False,
            rod=None,
            loc=None,
            fx=None,
            fy=None,
            world_to_camera=None
        )
    # print(f"the recovered_R=\n{recovered_R}")
    s = sKBI[2, 0] / recovered_R[2, 0]
    also_s = sKBI[2, 1] / recovered_R[2, 1]
    print(f"s={s}")
    print(f"also_s={also_s}")
    fx = 1 / (a * s * conditioning_factor)
    fy = 1 / (b * s * conditioning_factor)
    print(f"fx={fx}, fy={fy}")
    recovered_T = np.zeros(3)
    recovered_T[0] = sKBI[0, 2] / (fx * s) 
    recovered_T[1] = sKBI[1, 2] / (fy * s)
    recovered_T[2] = sKBI[2, 2] / s
    recovered_loc = - recovered_R.T @ recovered_T
    recovered_rod = rodrigues_from_SO3(recovered_R)
    return dict(
        success=True,
        rod=recovered_rod,
        loc=recovered_loc,
        fx=fx,
        fy=fy,
        world_to_camera=recovered_R
    )
