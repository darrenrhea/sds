from map_points_through_curvy_homography import (
     map_points_through_curvy_homography
)
import numpy as np


def report_curvy_homography_error(
    matched_source_points: np.ndarray,
    matched_destination_points: np.ndarray,
    numerator_params: np.ndarray,
    denominator_params: np.ndarray,
    names=None
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
    
    num_matches = matched_source_points.shape[0]

    predictions = map_points_through_curvy_homography(
        points=matched_source_points, 
        numerator_params=numerator_params,
        denominator_params=denominator_params,
    )
    residuals = predictions - matched_destination_points
    l2 = np.sum(np.abs(residuals**2), axis=1)
    

    if names is None:
        names = ["the point" for _ in range(num_matches)]


    for source_point, prediction, destination_point, point_name in zip(matched_source_points, predictions, matched_destination_points, names):
        print(f"The homography sends {point_name} in the ad they sent us, i.e. {source_point[0]:.1f}, {source_point[1]:.1f}")
        print(f"to {prediction[0]:.1f}, {prediction[1]:.1f} in the video frame")
        print(f"Whereas the 2D-2D data suggests it was supposed to hit     {destination_point[0]:.1f}, {destination_point[1]:.1f}")
        dist = np.linalg.norm(prediction - destination_point)
        print(f"A reprojection error of {dist=}")
        print("\n\n\n\n")


    return l2

