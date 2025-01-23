import numpy as np
from scipy.ndimage import map_coordinates
from homography_utils import (
    map_points_through_homography,
    solve_for_homography_based_on_4_point_correspondences
)


def quadrilateral_sampling(
    is_over_js: np.ndarray,
    t: float,
    constants: dict
) -> np.ndarray:
    """
    This is a warp function,
    i.e. given the (i, j) coordinates of a pixel in the output image / patch,
    this says where to lookup in the fullsize image
    """
    patch_height = constants["patch_height"] 
    patch_width = constants["patch_width"]

    patch_top_left_ij = np.array([0, 0], dtype=np.float32)
    patch_top_right_ij = np.array([0, patch_width - 1], dtype=np.float32)
    patch_bottom_right_ij = np.array([patch_height - 1, patch_width - 1], dtype=np.float32)
    patch_bottom_left_ij = np.array([patch_height - 1, 0], dtype=np.float32)

    fullsize_top_left_ij = np.array(constants['top_left_xy'][::-1], dtype=np.float32)
    fullsize_top_right_ij = np.array(constants['top_right_xy'][::-1], dtype=np.float32)
    fullsize_bottom_right_ij = np.array(constants['bottom_right_xy'][::-1], dtype=np.float32)
    fullsize_bottom_left_ij = np.array(constants['bottom_left_xy'][::-1], dtype=np.float32)

    for point in [
        patch_top_left_ij,
        patch_top_right_ij,
        patch_bottom_right_ij,
        patch_bottom_left_ij,
        fullsize_top_left_ij,
        fullsize_top_right_ij,
        fullsize_bottom_right_ij,
        fullsize_bottom_left_ij
    ]:
        assert isinstance(point, np.ndarray)
        assert point.shape == (2, )

    points_in_image_a = np.stack(
        [
            patch_top_left_ij,
            patch_top_right_ij,
            patch_bottom_right_ij,
            patch_bottom_left_ij
        ],
        axis=0
    )
    # print(f"{points_in_image_a=}")
    assert points_in_image_a.shape == (4, 2), f"{points_in_image_a.shape=} but must be shape (4, 2)"

    points_in_image_b = np.stack(
        [
            fullsize_top_left_ij,
            fullsize_top_right_ij,
            fullsize_bottom_right_ij,
            fullsize_bottom_left_ij
        ],
        axis=0
    )
    # print(f"{points_in_image_b=}")
    assert points_in_image_b.shape == (4, 2), f"{points_in_image_b.shape=} but must be shape (4, 2)"


    H = solve_for_homography_based_on_4_point_correspondences(
        points_in_image_a=points_in_image_a,
        points_in_image_b=points_in_image_b
    )

    assert is_over_js.shape == (2, patch_height * patch_width)

    mapped_points = map_points_through_homography(
        points=is_over_js,
        homography=H,
        a_point_is_a_row=False
    )
    assert mapped_points.shape == is_over_js.shape
    # every column is a point:
    assert mapped_points.shape == (2, patch_height * patch_width)
    return mapped_points
   

def make_is_over_js(
    patch_height: int,
    patch_width: int
) -> np.ndarray:
    """
    This makes is_over_js, which might be expensive, like 2.5 ms.
    All (i row, j col) locations within a patch_height x patch_width matrix are listed exactly once as a column of is_over_js.
    We make a 2 x (patch_height * patch_width) array,
    i.e. a matrix with 2 rows and patch_height * patch_width columns,
    The first row contains all the i values, i.e. row indices, the second row contains all the j indices.
    """
    # start_time = time.time()
    j_linspace = np.linspace(0, patch_width - 1, patch_width)
    assert j_linspace.shape == (patch_width, )

    i_linspace = np.linspace(0, patch_height - 1, patch_height)
    assert i_linspace.shape == (patch_height, )

    # either of these two lines will work:
    # two_dim_matrix_of_js, two_dim_matrix_of_is  = np.meshgrid(j_linspace, i_linspace, indexing='xy')
    two_dim_matrix_of_is, two_dim_matrix_of_js  = np.meshgrid(i_linspace, j_linspace, indexing='ij')
   
    assert two_dim_matrix_of_js.shape == (patch_height, patch_width)
    assert two_dim_matrix_of_is.shape == (patch_height, patch_width)
    is_over_js = np.stack(
        [
            two_dim_matrix_of_is.ravel(),
            two_dim_matrix_of_js.ravel()
        ],
        axis=0
    )
    assert is_over_js.shape == (2, patch_height * patch_width)
    # stop_time = time.time()
    # elapsed_time = stop_time - start_time
    # print(f"{elapsed_time=}")
    return is_over_js


def cutout_a_warped_patch_from_a_fullsize_image(
    patch_width: int,
    patch_height: int,
    fullsize_image_np_u8: np.ndarray,
    is_over_js: np.ndarray  # we have people pass this in so we don't have to make it over and over again
):
    """
    If fullsize_image_np_u8 is a numpy array of shape [H, W, C] where C is usually 3 or 4 or 5 (weight_mask is the 5th / 4-ith channel),
    Then this function cuts out a warped patch of size patch_height x patch_width from fullsize_image_np_u8
    by first cutting out a random quadrilateral and then warping it to a rectangle of that patch_size by homography.
    If fullsize_image_np_u8 is the patch_size, then this function just returns fullsize_image_np_u8.
    """
    if fullsize_image_np_u8.shape[0] == patch_height and fullsize_image_np_u8.shape[1] == patch_width:
        # honestly, we could warp in this situation as well,
        # like pick a quadralateral slightly inside 1920x1080
        # and homographically warp it to 1920x1088
        # Today we choose not to.
        # print(f"{Fore.YELLOW}WARNING: no warping going on{Style.RESET_ALL}")
        warped_np_u8 = fullsize_image_np_u8
        assert warped_np_u8.shape == (patch_height, patch_width, fullsize_image_np_u8.shape[2])
        assert warped_np_u8.dtype == np.uint8
        return warped_np_u8

    while True:
        warp_function = quadrilateral_sampling
        t = 0  # no temporal variation of warping
        # random zoom factor:
        zoom_factor = np.random.uniform(0.9, 1.7)
        sw = zoom_factor * patch_width
        sh = zoom_factor * patch_height
        x0 = np.random.uniform(low=0, high=fullsize_image_np_u8.shape[1] - sw)
        y0 = np.random.uniform(low=0, high=fullsize_image_np_u8.shape[0] - sh)
        # x0 = 500
        # y0 = 250
        x1 = x0 + sw
        y1 = y0 + sh

        # start with a nice axis aligned subrectangle of the fullsize image:
        orthogonal_top_left_xy = np.array([x0, y0], dtype=np.float32)
        orthogonal_top_right_xy = np.array([x1, y0], dtype=np.float32)
        orthogonal_bottom_right_xy = np.array([x1, y1], dtype=np.float32)
        orthogonal_bottom_left_xy = np.array([x0, y1], dtype=np.float32)

        # random deviation from orthogonal:
        std = min(patch_width, patch_height) * 0.075
        top_left_xy = orthogonal_top_left_xy + np.random.normal(0, std, size=(2, ))
        top_right_xy = orthogonal_top_right_xy + np.random.normal(0, std, size=(2, ))
        bottom_right_xy = orthogonal_bottom_right_xy + np.random.normal(0, std, size=(2, ))
        bottom_left_xy = orthogonal_bottom_left_xy + np.random.normal(0, std, size=(2, ))

        within_bounds = True
        for point_xy in [top_left_xy, top_right_xy, bottom_right_xy, bottom_left_xy]:
            if point_xy[0] < 0:
                within_bounds = False
                break
            if point_xy[0] > fullsize_image_np_u8.shape[1] - 1:
                within_bounds = False
                break
            if point_xy[1] < 0:
                within_bounds = False
                break
            if point_xy[1] > fullsize_image_np_u8.shape[0] - 1:
                within_bounds = False
                break
        
        # sometimes the random quadrilateral is outside the image:
        if not within_bounds:
            continue  # try again
        else:
            break
    
    # having found a quadrilateral that fits inside the fullsize image, cut it out:
    constants = dict(
        patch_height=patch_height,
        patch_width=patch_width,
        top_left_xy=top_left_xy,
        top_right_xy=top_right_xy,
        bottom_left_xy=bottom_left_xy,
        bottom_right_xy=bottom_right_xy
    )

    warped_is_over_js = warp_function(
        is_over_js=is_over_js,
        t=t,
        constants=constants
    )
    assert warped_is_over_js.shape == (2, patch_height * patch_width), f"{warped_is_over_js.shape=}"

    warped_f32 = np.zeros((patch_height, patch_width, fullsize_image_np_u8.shape[2]), dtype=np.float32)

    for c in range(0, fullsize_image_np_u8.shape[2]):
        
        warped_f32[:, :, c] = map_coordinates(
            input=fullsize_image_np_u8[:, :, c],
            coordinates=warped_is_over_js,  # [row of fractional is_over a row of fractional js]
            order=1,
            cval=0  #  for alpha channel this is right extension value for off the map.
        ).reshape((patch_height, patch_width))

    warped_np_u8 = np.round(warped_f32).clip(0, 255).astype(np.uint8)
    
    assert warped_np_u8.shape == (patch_height, patch_width, fullsize_image_np_u8.shape[2])
    assert warped_np_u8.dtype == np.uint8
    return warped_np_u8


def cutout_warped_patches_from_fullsize(
    patch_width: int,
    patch_height: int,
    num_patches_to_generate: int,
    fullsize_image_np_u8: np.ndarray,
):

    """
    Given a full-size image and 4 points in [0, 1080) x [0, 1920)
    that are the vertices of a quadrilateral,
    cut out that quadrilateral and warp it to a rectangle
    of size patch_width x patch_height.
    """
    num_channels = fullsize_image_np_u8.shape[2]
    assert isinstance(patch_width, int), "ERROR: patch_width must be an integer"
    assert isinstance(patch_height, int), "ERROR: patch_height must be an integer"
    assert isinstance(num_patches_to_generate, int), "ERROR: num_patches_to_generate must be an integer"
    assert patch_width > 0, "ERROR: patch_width must be positive"
    assert patch_height > 0, "ERROR: patch_height must be positive"
    assert num_patches_to_generate >= 0, "ERROR: num_patches_to_generate must be non-negative?"

    assert isinstance(fullsize_image_np_u8, np.ndarray), "ERROR: fullsize_image_np_u8 must be a numpy array"
    assert fullsize_image_np_u8.ndim == 3, "ERROR: fullsize_image_np_u8 must be a numpy array of shape [H, W, C] where C is usually 3"
    assert fullsize_image_np_u8.shape[2] in  [3, 4], "ERROR: fullsize_image_np_u8 must be a numpy array of shape [H, W, C] where C is usually 3 or 4"

    is_over_js = make_is_over_js(patch_height=patch_height, patch_width=patch_width)

    patches = np.zeros((num_patches_to_generate, patch_height, patch_width, num_channels), dtype=np.uint8)
    num_generated = 0

    while True:
        if num_generated >= num_patches_to_generate:
            break
        
        warped_np_u8 = cutout_a_warped_patch_from_a_fullsize_image(
            patch_width=patch_width,
            patch_height=patch_height,
            fullsize_image_np_u8=fullsize_image_np_u8,
            is_over_js=is_over_js
        )
     

        patches[num_generated, ...] = warped_np_u8        
        num_generated += 1
    
    assert isinstance(patches, np.ndarray)
    assert patches.ndim == 4
    assert patches.shape[0] == num_patches_to_generate
    assert patches.shape[1] == patch_height
    assert patches.shape[2] == patch_width
    assert patches.shape[3] == fullsize_image_np_u8.shape[2]
    return patches


