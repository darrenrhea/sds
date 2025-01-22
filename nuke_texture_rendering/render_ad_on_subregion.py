import time
import numpy as np
from scipy.ndimage import map_coordinates
import rodrigues_utils
from CameraParameters import CameraParameters


def render_ad_on_subregion(
    ad_origin: np.ndarray,  # the origin of the ad in world coordinates, often we choose the bottom left corner
    u: np.ndarray, # the to-the-right unit vector in the ad's plane in world coordinates
    v: np.ndarray, # the "up" unit vector in the ad's plane in world coordinates.  Some LED boards tilt back, so this is not necessarily the same as the world's up vector
    photograph_width_in_pixels: int,
    photograph_height_in_pixels: int,
    camera_parameters: CameraParameters,
    texture_rgba_np_float32: np.ndarray,  # the texture to use as the ad
    ad_width,
    ad_height,
    # suppose you know a list of pixels that could possibly be affected:
    ijs: np.ndarray # a list of pixel locations canvas that you want to render
) -> np.ndarray:
    """
    Here *rendering* means maybe starting with an actual photograph and then
    adding an augmented reality ad onto a plane in the 3D world.
    according to the Nuke lens model camera solve in camera_parameters.

    Even and especially if you aren't inserting the ad into an actual photograph but just a blackness, you
    must still specify photograph_width_in_pixels and photograph_height_in_pixels so that we
    know how big a full render would be.

    Ordinarily one renders the full-sized image, which might be like 1920x1080 or 3840x2160 or something.
    But, you may want to render only a small subrectangle of the entire rendering for various purposes, including
        * using the good camera solve on the current frame to make Blender-like patch templates to search for in the next frame
        * speed (it is faster to render less)
        * zooming in (to see how well something lines up, you only need to look closely at that area)
     
    Give a texture to place on the floor in the world-coordintes spot
    [0, ad_width] x [0, ad_height] x {0}

    instead of returning a full-sized rendered image, it returns just full_image[i_min:i_max, j_min, j_max]
    as a np rbga array.
    """
    assert isinstance(ijs, np.ndarray)
    assert ijs.ndim == 2
    assert ijs.shape[1] == 2
    assert ijs.dtype == np.int64

    assert isinstance(ad_origin, np.ndarray)
    assert ad_origin.shape == (3,)
    assert ad_origin.dtype == np.float64 or ad_origin.dtype == np.float32

    assert isinstance(u, np.ndarray)
    assert u.shape == (3,)
    assert u.dtype == np.float64 or u.dtype == np.float32
    assert np.isclose(np.linalg.norm(u), 1)

    assert isinstance(v, np.ndarray)
    assert v.shape == (3,)
    assert v.dtype == np.float64 or v.dtype == np.float32
    assert np.isclose(np.linalg.norm(v), 1)




    assert isinstance(camera_parameters, CameraParameters)

    cameras_location_in_world_coordinates = camera_parameters.loc
    rodrigues_vector = camera_parameters.rod
    focal_length = camera_parameters.f
    k1 = camera_parameters.k1
    k2 = camera_parameters.k2
    k3 = camera_parameters.k3

    ppi = camera_parameters.ppi
    ppj = camera_parameters.ppj
    p1 = camera_parameters.p1
    p2 = camera_parameters.p2

    assert (
        texture_rgba_np_float32.shape[2] == 4
    ), f"texture_rgba_np_float32 should br RGBA and thus must have 4 channels not {texture_rgba_np_float32.shape}"

    world_to_camera = rodrigues_utils.SO3_from_rodrigues(
        rodrigues_vector
    )

    # unpack the rows into the camera's axes:
    cameras_x_axis_in_wc, cameras_y_axis_in_wc, cameras_z_axis_in_wc = world_to_camera


    aspect_ratio_less_than_one = photograph_height_in_pixels / photograph_width_in_pixels

    assert aspect_ratio_less_than_one < 1

    # now we have the normalized coordinates:
    xd = ijs[:, 1].astype(np.float32) / (photograph_width_in_pixels - 1.0) * 2 - 1
    yd = (ijs[:, 0].astype(np.float32) / (photograph_height_in_pixels - 1.0) * 2 - 1) * aspect_ratio_less_than_one

    x = xd - ppj
    y = yd - ppi
    xdf = x / focal_length
    ydf = y / focal_length
    # undistort it:
    r2 = xdf ** 2 + ydf ** 2
    c = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2**3
    
    # the point on the image plane in camera coordinates is (vx_cc, vy_cc, vz_cc)
    vx_cc = c * x + p1 * (r2 + 2 * x**2) + 2 * p2 * x * y # this undistorts it. Note we do not add ppj on purpose
    vy_cc = c * y + p2 * (r2 + 2 * y**2) + 2 * p1 * x * y # this undistorts it. Note we do not add ppi on purpose
    vz_cc = focal_length

    vx_wc = (
        cameras_x_axis_in_wc[0] * vx_cc
        + cameras_y_axis_in_wc[0] * vy_cc
        + cameras_z_axis_in_wc[0] * vz_cc
    )
    vy_wc = (
        cameras_x_axis_in_wc[1] * vx_cc
        + cameras_y_axis_in_wc[1] * vy_cc
        + cameras_z_axis_in_wc[1] * vz_cc
    )
    vz_wc = (
        cameras_x_axis_in_wc[2] * vx_cc
        + cameras_y_axis_in_wc[2] * vy_cc
        + cameras_z_axis_in_wc[2] * vz_cc
    )

   

    # Suppose we are hitting the plane spanned by u, v starting at "ad_origin":
    # normal = np.cross(u, v)
    # plane is the locus of x such that np.dot(x - origin, normal) = 0
    # x dot normal = origin dot normal
    # so we want to solve for t the equation np.dot(cameras_location_in_world_coordinates + t * velocity_wc, normal) =  origin dot normal
    # so t = <origin | normal> - <cameras_location_in_world_coordinates| normal> / <velocity_wc| normal>
    
    normal = np.cross(u, v)
    normal_x, normal_y, normal_z = normal


    numerator = np.dot(ad_origin - cameras_location_in_world_coordinates, normal)
    
    assert isinstance(numerator, float), f"numerator should be a float but is {type(numerator)=}"

    denom = vx_wc * normal_x + vy_wc * normal_y + vz_wc * normal_z

    t_hit = numerator / denom
    
    x_hit = cameras_location_in_world_coordinates[0] + t_hit * vx_wc
    y_hit = cameras_location_in_world_coordinates[1] + t_hit * vy_wc
    z_hit = cameras_location_in_world_coordinates[2] + t_hit * vz_wc

    x_hit -= ad_origin[0]
    y_hit -= ad_origin[1]
    z_hit -= ad_origin[2]

    # if hit ad_origin + m u + n v, what are m and n?
    m_hit = x_hit * u[0] + y_hit * u[1] + z_hit * u[2]
    n_hit = x_hit * v[0] + y_hit * v[1] + z_hit * v[2]

    j_within_ad_raster = (
        (m_hit - 0)
        / ad_width
        * (texture_rgba_np_float32.shape[1] - 1)
    )
    i_within_ad_raster = (
        (ad_height - n_hit)
        / ad_height
        * (texture_rgba_np_float32.shape[0] - 1)
    )

    rgba_values = np.zeros((ijs.shape[0], 4), dtype=np.float32)
    start_time = time.time()
    for c in range(0, 4):
        rgba_values[:, c] = map_coordinates(
            input=texture_rgba_np_float32[:, :, c],
            coordinates=[i_within_ad_raster, j_within_ad_raster],
            order=1,
            cval=0,
        )
    stop_time = time.time()
    duration = stop_time - start_time
    print(f"map_coordinates took {duration} seconds")
    return rgba_values

