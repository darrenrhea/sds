"""
Renders a 3D image of the floor texture under a given set of camera parameters.
Not only can it render a full-sized 3d image,
but importantly it can partially render, i.e.
what would a given subrectangle of the full render look like.
"""
import numpy as np
import PIL
import PIL.Image
from scipy.ndimage import map_coordinates
import rodrigues_utils
from CameraParameters import CameraParameters

def partial_render(
    photograph_width_in_pixels: int,
    photograph_height_in_pixels: int,
    original_photo_rgba_np_uint8: np.ndarray,
    camera_parameters: CameraParameters,
    texture_rgba_np_float32: np.ndarray,
    texture_x_min_wc,
    texture_x_max_wc,
    texture_y_min_wc,
    texture_y_max_wc,
    i_min,
    i_max,
    j_min,
    j_max,
):
    """
    See full_nuke_texture_render for non-partial rendering.
    Here *rendering* means maybe starting with an actual photograph and then
    adding an augmented reality floor texture into the floor z=0
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
    [texture_x_min_wc, texture_x_max_wc] x [texture_y_min_wc, texture_y_max_wc] x {0}

    instead of returning a full-sized rendered image, it returns just full_image[i_min:i_max, j_min, j_max]
    as a np rbga array.
    """
    if original_photo_rgba_np_uint8 is not None:
        assert (
            original_photo_rgba_np_uint8.shape[2] == 4
        ), "original_photo_rgba_np_uint8 should be RGBA thus 4 channels"
        assert (
            original_photo_rgba_np_uint8.shape[0] == photograph_height_in_pixels
        ), "original_photo_rgba_np_uint8 must have same size as the render"
        assert (
            original_photo_rgba_np_uint8.shape[1] == photograph_width_in_pixels
        ), "original_photo_rgba_np_uint8 must have same size as the render"

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

    h_image = photograph_height_in_pixels
    w_image = photograph_width_in_pixels
    # sometimes we want to draw on top of an image:
    if original_photo_rgba_np_uint8 is not None:
        image_np_uint8 = original_photo_rgba_np_uint8[i_min:i_max, j_min:j_max, :]
    else:
        image_np_uint8 = np.zeros(
            shape=(i_max - i_min, j_max - j_min, 4), dtype=np.uint8
        )

    x_linspace = np.linspace(-1, 1, w_image)[j_min:j_max]
    assert x_linspace.shape == (j_max - j_min,)

    aspect_ratio_less_than_one = photograph_height_in_pixels / photograph_width_in_pixels

    assert aspect_ratio_less_than_one < 1

    y_linspace = np.linspace(
        -aspect_ratio_less_than_one,
        aspect_ratio_less_than_one,
        h_image
    )[i_min:i_max]
    
    assert y_linspace.shape == (i_max - i_min,)

    xd, yd = np.meshgrid(x_linspace, y_linspace)
    assert xd.shape == (i_max - i_min, j_max - j_min)
    assert yd.shape == (i_max - i_min, j_max - j_min)
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
    ad_origin = np.array(
        [0, 0, 0]
    )
    u = np.array([1, 0, 0])
    v = np.array([0, 0, 1])
    normal = np.cross(u, v)
    normal_x, normal_y, normal_z = normal


    numerator = np.dot(ad_origin - cameras_location_in_world_coordinates, normal)
    
    assert isinstance(numerator, float), f"numerator should be a float but is {type(numerator)=}"

    denom = vx_wc * normal_x + vy_wc * normal_y + vz_wc * normal_z

    t_hit = numerator / denom
    print(f"{t_hit.shape=}")


    # t_hit = -cameras_location_in_world_coordinates[2] / vz_wc  # when it hits the floor
    # print(f"{t_hit.shape=}")
    x_hit = cameras_location_in_world_coordinates[0] + t_hit * vx_wc
    y_hit = cameras_location_in_world_coordinates[1] + t_hit * vy_wc
    z_hit = cameras_location_in_world_coordinates[2] + t_hit * vz_wc

    x_hit -= ad_origin[0]
    y_hit -= ad_origin[1]
    z_hit -= ad_origin[2]

    # if hit ad_origin + m u + n v, what are m and n?
    m_hit = x_hit * u[0] + y_hit * u[1] + z_hit * u[2]
    n_hit = x_hit * v[0] + y_hit * v[1] + z_hit * v[2]

    assert texture_x_min_wc == 0
    assert texture_y_min_wc == 0

    j_within_ad_raster = (
        (m_hit - texture_x_min_wc)
        / (texture_x_max_wc - texture_x_min_wc)
        * (texture_rgba_np_float32.shape[1] - 1)
    )
    i_within_ad_raster = (
        (texture_y_max_wc - n_hit)
        / (texture_y_max_wc - texture_y_min_wc)
        * (texture_rgba_np_float32.shape[0] - 1)
    )

    # we move into PIL.Images since alpha compositing is defined therein:
    if original_photo_rgba_np_uint8 is None:
        total_alpha_composited_pil = PIL.Image.fromarray(
            np.zeros(shape=(i_max - i_min, j_max - j_min, 4), dtype=np.uint8)
        )
    else:
        total_alpha_composited_pil = PIL.Image.fromarray(
            original_photo_rgba_np_uint8[i_min:i_max, j_min:j_max, :]
        )

    # could for loop over multiple insertions:
    ads_contribution_rgba_np_float32 = np.zeros(
        shape=(i_max - i_min, j_max - j_min, 4), dtype=np.float32
    )

    for c in range(0, 4):
        ads_contribution_rgba_np_float32[:, :, c] = map_coordinates(
            input=texture_rgba_np_float32[:, :, c],
            coordinates=[i_within_ad_raster, j_within_ad_raster],
            order=1,
            cval=0,
        )

    ads_contribution_rgba_np_uint8 = ads_contribution_rgba_np_float32.clip(
        0, 255
    ).astype(np.uint8)
    ads_contribution_pil = PIL.Image.fromarray(ads_contribution_rgba_np_uint8)

    total_alpha_composited_pil = PIL.Image.alpha_composite(
        im1=total_alpha_composited_pil,  # must be mode RGBA
        im2=ads_contribution_pil  # must be mode RGBA. This one is on top
    )

    # where = (ad_at_full_resolution_uint8[:,:,3] == 255)
    # image_np_uint8[where, :] = ads_contribution_rgba_np_float32[where, :]

    final_rga_np_uint8 = np.array(total_alpha_composited_pil)
    return final_rga_np_uint8

