from least_positive_element_and_index_over_last_axis import (
     least_positive_element_and_index_over_last_axis
)
from get_t_hit import (
     get_t_hit
)
from typing import List
from AdPlacementDescriptor import (
     AdPlacementDescriptor
)
import numpy as np
from scipy.ndimage import map_coordinates
import rodrigues_utils
from CameraParameters import CameraParameters


def render_ads_on_subregion(
    ad_placement_descriptors: List[AdPlacementDescriptor],
    photograph_width_in_pixels: int,
    photograph_height_in_pixels: int,
    camera_parameters: CameraParameters,
    ijs: np.ndarray # a list of pixel locations on the canvas that you want to render
) -> np.ndarray:
    """
    Ordinarily you might think about rendering several textured quads in 3D
    as a full-sized image, which might be like 1920x1080 or 3840x2160 or something.

    But, you may want to render only a small subset of entire rendering for various purposes,
    especially for speed.


    ijs is a list of (i, j) (row, column) coordinates of which pixels you want to render.
    This function will return a list of RGBA values for those pixel locations.


    Allows for multiple quads, where one might be infront of another.
    The first hitting time wins.
    
    Uses the Nuke lens model.

    You must still specify photograph_width_in_pixels and photograph_height_in_pixels so that we
    know how big a full render would be.

   
    """
    # if nothing to render, return the correct number of completely transparent RGBA values:
    if len(ad_placement_descriptors) == 0:
        return np.zeros((ijs.shape[0], 4), dtype=np.float32)
    
    assert isinstance(ijs, np.ndarray)
    assert ijs.ndim == 2
    assert ijs.shape[1] == 2, "ijs should have two columns"
    assert ijs.dtype == np.int64, "currently only int64 is supported, prob will move to floats"
    num_samples = ijs.shape[0]
    num_ads = len(ad_placement_descriptors)

    # BEGIN check ad_placement_descriptors:
    for ad_placement_descriptor in ad_placement_descriptors:
        ad_origin = ad_placement_descriptor.origin
        u = ad_placement_descriptor.u
        v = ad_placement_descriptor.v
        ad_height = ad_placement_descriptor.height
        ad_width = ad_placement_descriptor.width
        
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

        assert isinstance(ad_height, float)
        assert ad_width > 0
        assert isinstance(ad_width, float)
        assert ad_height > 0
        texture_rgba_np_f32 = ad_placement_descriptor.texture_rgba_np_f32
        assert (
            texture_rgba_np_f32.shape[2] == 4
        ), f"texture_rgba_np_f32 should br RGBA and thus must have 4 channels not {texture_rgba_np_f32.shape}"

    #ENDOF check ad_placement_descriptors.
        
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

  
    world_to_camera = rodrigues_utils.SO3_from_rodrigues(
        rodrigues_vector
    )

    # BEGIN calculate world velocities of the backwards light rays:
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
    vx_cc = c * x #  p1 * (r2 + 2 * x**2) + 2 * p2 * x * y # this undistorts it. Note we do not add ppj on purpose
    vy_cc = c * y #  p2 * (r2 + 2 * y**2) + 2 * p1 * x * y # this undistorts it. Note we do not add ppi on purpose
    vz_cc = focal_length

    # alternative explanation that is equivalent:
    # vx_cc_alt = c * xdf = c * y / focal_length
    # vy_cc_alt = c * ydf = c * y / focal_length
    # vz_cc_alt = 1 = focal_length / focal_length

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

    # ENDOF calculate world velocities of the backwards light rays.
    t_hits = np.zeros(
        shape=(num_samples, num_ads),
        dtype=np.float32
    )

    for ad_index, ad_placement_descriptor in enumerate(ad_placement_descriptors):
        t_hit = get_t_hit(
            ad_placement_descriptor=ad_placement_descriptor,
            camera_parameters=camera_parameters,
            vx_wc=vx_wc,
            vy_wc=vy_wc,
            vz_wc=vz_wc,
        )
        t_hits[:, ad_index] = t_hit

    x_hits = cameras_location_in_world_coordinates[0] + t_hits * vx_wc[:, np.newaxis]
    y_hits = cameras_location_in_world_coordinates[1] + t_hits * vy_wc[:, np.newaxis]
    z_hits = cameras_location_in_world_coordinates[2] + t_hits * vz_wc[:, np.newaxis]
    
    assert (
        x_hits.shape == (num_samples, num_ads)
    )

    ad_origin_xs = np.array(
        [ad_placement_descriptor.origin[0] for ad_placement_descriptor in ad_placement_descriptors]
    )
    ad_origin_ys = np.array(
        [ad_placement_descriptor.origin[1] for ad_placement_descriptor in ad_placement_descriptors]
    )
    ad_origin_zs = np.array(
        [ad_placement_descriptor.origin[2] for ad_placement_descriptor in ad_placement_descriptors]
    )

    x_hits -= ad_origin_xs
    y_hits -= ad_origin_ys
    z_hits -= ad_origin_zs

    assert (
        x_hits.shape == (num_samples, num_ads)
    )

    ad_u_xs = np.array(
        [ad_placement_descriptor.u[0] for ad_placement_descriptor in ad_placement_descriptors]
    )
    ad_u_ys = np.array(
        [ad_placement_descriptor.u[1] for ad_placement_descriptor in ad_placement_descriptors]
    )
    ad_u_zs = np.array(
        [ad_placement_descriptor.u[2] for ad_placement_descriptor in ad_placement_descriptors]
    )
    ad_v_xs = np.array(
        [ad_placement_descriptor.v[0] for ad_placement_descriptor in ad_placement_descriptors]
    )
    ad_v_ys = np.array(
        [ad_placement_descriptor.v[1] for ad_placement_descriptor in ad_placement_descriptors]
    )
    ad_v_zs = np.array(
        [ad_placement_descriptor.v[2] for ad_placement_descriptor in ad_placement_descriptors]
    )
    assert (
        ad_u_xs.shape == (num_ads,)
    )
    # Basically if hit = ad_origin + m u + n v, what are m and n?
    # m = u dotproduct (m u + n v) only works if u dot v = 0
    # n = v dotproduct (m u + n v) only works if u dot v = 0
    # Orthonormal vectors make there own inverse, in the sense that (uT over vT) matrix mult (u | v) = 1
    # by stacking u an v as columns into a 2x2 matrix, we can calulate the inverse of that matrix and then the rows of the inverse are the extractor vectors:
    # so say you know the x, y, and z coordinates m u + n v = (x, y, z).
    # what is a formula for m and n that works for any u and v, including non-orthonormal u and v?
    # m u + n v + p w = (x, y, z)
    # [u | v | w] * [m n p]^T = [x y z]^T

    m_hits = ad_u_xs * x_hits + ad_u_ys * y_hits + ad_u_zs * z_hits
    n_hits = ad_v_xs * x_hits + ad_v_ys * y_hits + ad_v_zs * z_hits
    
    assert m_hits.shape == (num_samples, num_ads)

    ad_widths = np.array(
        [ad_placement_descriptor.width for ad_placement_descriptor in ad_placement_descriptors]
    )
    ad_heights = np.array(
        [ad_placement_descriptor.height for ad_placement_descriptor in ad_placement_descriptors]
    )
    
    texture_int_heights = np.array(
        [ad_placement_descriptor.texture_rgba_np_f32.shape[0] for ad_placement_descriptor in ad_placement_descriptors]
    )
    
    texture_int_widths = np.array(
        [ad_placement_descriptor.texture_rgba_np_f32.shape[1] for ad_placement_descriptor in ad_placement_descriptors]
    )
   
    junit_within_ad_raster = (
        (m_hits - 0)
        / ad_widths
    )
              
    assert junit_within_ad_raster.shape == (num_samples, num_ads)


    iunit_within_ad_raster = (
        (ad_heights - n_hits)
        / ad_heights
    )
    in_rectangle = (
        (junit_within_ad_raster > 0)
        *
        (junit_within_ad_raster < 1)
        *
        (iunit_within_ad_raster > 0)
        *
        (iunit_within_ad_raster < 1)
    )
    t_hits[~in_rectangle] = -np.inf
    
    t_min, which_ad = least_positive_element_and_index_over_last_axis(t_hits)

    assert t_min.shape == (num_samples,)

    assert(
        junit_within_ad_raster.shape == (num_samples, num_ads)
    )
    assert texture_int_widths.shape == (num_ads,)
    assert which_ad.shape == (num_samples,)
    # starting to need to know who won.
   
    indices = np.array(range(0, num_samples), dtype=np.int64)
  
    j_within_ad_raster = (
        junit_within_ad_raster[indices, which_ad] * (texture_int_widths[which_ad] - 1)
    )
    

    i_within_ad_raster = (
        iunit_within_ad_raster[indices, which_ad]
        * (texture_int_heights[which_ad] - 1)
    )

    rgba_values = np.zeros((ijs.shape[0], 4), dtype=np.float32)
    for ad_index in range(0, num_ads):
        texture_rgba_np_f32 = ad_placement_descriptors[ad_index].texture_rgba_np_f32
        i_s = i_within_ad_raster[which_ad == ad_index]
        j_s = j_within_ad_raster[which_ad == ad_index]
        for c in range(0, 4):
            rgba_values[which_ad == ad_index, c] = map_coordinates(
                input=texture_rgba_np_f32[:, :, c],
                coordinates=[i_s, j_s],
                order=1,
                cval=0,
            )
    return rgba_values

