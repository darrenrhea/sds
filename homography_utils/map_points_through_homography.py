import numpy as np
import cv2


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
    
    assert (
        points.dtype == np.float32
        or
        points.dtype == np.float64
    ), f"points must be dtype np.float32 or np.float64 but it is {points.dtype=}"
    
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
