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
from homography_utils import (
    map_points_through_homography,
    solve_for_homography_based_on_4_point_correspondences
)

def homography_ad_insertion(
    photograph_width_in_pixels,
    photograph_height_in_pixels,
    original_photo_rgba_np_uint8,
    texture_rgba_np_float32,
    homography_from_photo_to_world_floor,
    texture_x_min_wc,
    texture_x_max_wc,
    texture_y_min_wc,
    texture_y_max_wc,
):
    """
    Starting with an actual photograph and then
    adding an augmented reality floor texture into the floor z=0
    according to the homography_from_photo_to_texture.

    Even and especially if you aren't inserting the ad into an actual photograph but just blackness, you
    must specify photograph_width_in_pixels and photograph_height_in_pixels so that we
    know how big a full render would be.
    """
    texture_raster_height, texture_raster_width, _ = texture_rgba_np_float32.shape
    homography_from_texture_to_world_floor = solve_for_homography_based_on_4_point_correspondences(
        points_in_image_a=np.array(
            [
                [0, 0],
                [texture_raster_width, 0],
                [texture_raster_width, texture_raster_height],
                [0, texture_raster_height]
            ]
        ),
        points_in_image_b=np.array(
            [
                [texture_x_min_wc, texture_y_max_wc],
                [texture_x_max_wc, texture_y_max_wc],
                [texture_x_max_wc, texture_y_min_wc],
                [texture_x_min_wc, texture_y_min_wc]
            ]
        ),
    )
    homography_from_world_floor_to_texture = np.linalg.inv(homography_from_texture_to_world_floor)

    homography_from_photo_to_texture =  homography_from_world_floor_to_texture @ homography_from_photo_to_world_floor
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


    assert (
        texture_rgba_np_float32.shape[2] == 4
    ), f"texture_rgba_np_float32 should br RGBA and thus must have 4 channels not {texture_rgba_np_float32.shape}"


    h_image = photograph_height_in_pixels
    w_image = photograph_width_in_pixels
    if original_photo_rgba_np_uint8 is not None:
        image_np_uint8 = original_photo_rgba_np_uint8
    else:
        image_np_uint8 = np.zeros(
            shape=(photograph_height_in_pixels, photograph_width_in_pixels, 4), dtype=np.uint8
        )

    x_linspace = np.linspace(0, photograph_width_in_pixels - 1, photograph_width_in_pixels)
    y_linspace = np.linspace(0, photograph_height_in_pixels - 1, photograph_height_in_pixels)
    xd, yd = np.meshgrid(x_linspace, y_linspace)
    assert xd.shape == (photograph_height_in_pixels, photograph_width_in_pixels)
    assert yd.shape == (photograph_height_in_pixels, photograph_width_in_pixels)
    xd_ravel = xd.ravel()
    yd_ravel = yd.ravel()

    xy = np.column_stack([xd_ravel, yd_ravel])
    xy_in_texture = map_points_through_homography(
        inhomo_points_as_rows=xy,
        homography_3x3=homography_from_photo_to_texture,
    )

    j_within_ad_raster = xy_in_texture[:, 0].reshape((1080, 1920))
    i_within_ad_raster = xy_in_texture[:, 1].reshape((1080, 1920))

    # we move into PIL.Images since alpha compositing is defined therein:
    total_alpha_composited_pil = PIL.Image.fromarray(
        image_np_uint8
    )

    # could for loop over multiple insertions:
    ads_contribution_rgba_np_float32 = np.zeros(
        shape=(photograph_height_in_pixels, photograph_width_in_pixels, 4),
        dtype=np.float32
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
        total_alpha_composited_pil, ads_contribution_pil
    )

    photo_pixels = np.array(
        [
            [0, 0],
            [-47, 25],
        ]
    )
    final_rga_np_uint8 = np.array(total_alpha_composited_pil)
    
    # print(photo_pixels)
    # ans = map_points_through_homography(
    #     photo_pixels,
    #     np.linalg.inv(homography_from_photo_to_world_floor)
    # )
    # for x, y in ans:
    #     print(f"considering Drawing at x, y = {x}, {y}")
    #     if 2 < x and x < 1918 and 1 < y and y < 1079:
    #         x = int(x)
    #         y = int(y)
    #         print(f"Drawing at x, y = {x}, {y}")
    #         final_rga_np_uint8[y-1:y+1, x-1:x+1, :] = [255, 0, 0, 255]
    
    return final_rga_np_uint8

