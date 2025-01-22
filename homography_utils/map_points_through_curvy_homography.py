import numpy as np
import cv2


def map_points_through_curvy_homography(
    points: np.ndarray,  # a numpy array of 2D points (either every row is a point, so it is a N x 2 array, or every column is a point, so it is a 2 x N array)
    numerator_params,
    denominator_params,
) -> np.ndarray:
    """
    The use of cv2.perspectiveTransform is slightly strange
    in that you need an extra trivial dimension in the input.
    """
    assert points.ndim == 2
    assert points.shape[1] == 2, "num_columns must be 2 if a_point_is_a_row"    
    assert points.dtype == np.float32 or points.dtype == np.float64
   
    
    # numerator_params = [
    #         [M02, M00, M01, 0, 0, 0], # coefficients of 1 x y x^2 xy y^2 for the numerator of x_out
    #         [M12, M10, M11, 0, 0, 0], # coefficients of 1 x y x^2 xy y^2 for the numerator of y_out
    # ].T

    # denominator_params = torch.tensor(
    #     np.array([M22, M20, M21]).T,  # 1, x, y
    #     dtype=torch.float64,
    #     requires_grad=True
    # )
    
    N = points.shape[0]

    # sort of the design matrix of the situation:
    # columns are 1 x y x^2 xy y^2
    design_matrix = np.stack(
        [
            np.ones((N,), dtype=np.float64),
            points[:, 0],
            points[:, 1],
            points[:, 0] ** 2,
            points[:, 0] * points[:, 1],
            points[:, 1] ** 2,
        ],
        axis=1
    )
    print(f"{design_matrix=}")
    with_ones = np.stack(
        [
            np.ones((N,), dtype=np.float64),
            points[:, 0],
            points[:, 1],
        ],
        axis=1
    )
    print("with_ones:")
    print(with_ones)
    print("denominator_params:")
    print(denominator_params)

    print(f"{numerator_params=}")
    
    print(f"{design_matrix=}")

    numerators = design_matrix @ numerator_params
    denominator = with_ones @ denominator_params
    mapped_points = numerators / denominator[:, None]
    return mapped_points
