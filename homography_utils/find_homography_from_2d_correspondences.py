import numpy as np
import cv2


def find_homography_from_2d_correspondences(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    ransacReprojThreshold=0.5
):
    """
    Sometimes you already know 2D-2D matches and you want the least/low reprojection error
    homography that approximately explains it.
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
    """
    assert isinstance(src_points, np.ndarray)
    assert isinstance(dst_points, np.ndarray)
    assert src_points.ndim == 2
    assert dst_points.ndim == 2
    assert src_points.shape[0] == dst_points.shape[0]
    assert src_points.shape[1] == 2
    assert dst_points.shape[1] == 2
    assert src_points.dtype == np.float32 or src_points.dtype == np.float64
    assert dst_points.dtype == np.float32 or dst_points.dtype == np.float64

    srcPoints = src_points.astype(np.float32).reshape(-1, 1, 2)
    dstPoints = dst_points.astype(np.float32).reshape(-1, 1, 2)

    # homography_3x3, mask = cv2.findHomography(
    #     srcPoints=srcPoints,
    #     dstPoints=dstPoints,
    #     method=cv2.RANSAC,
    #     ransacReprojThreshold=ransacReprojThreshold
    # )
    homography_3x3, mask = cv2.findHomography(
        srcPoints=srcPoints,
        dstPoints=dstPoints,
        method=cv2.USAC_ACCURATE, #cv2.RANSAC,
        ransacReprojThreshold=ransacReprojThreshold
    )
    assert mask.shape[1] == 1  # we will get rid of this unnecessary dimension below
    assert mask.dtype == np.uint8  # we are going to cast this to np.bool

    match_indicator = mask.ravel().astype(bool)  # bool can work as an indicator
    matched_src_points = src_points[match_indicator]
    matched_dst_points = dst_points[match_indicator]
    
    return dict(
        success=True,
        homography_3x3=homography_3x3,
        matched_src_points=matched_src_points,
        matched_dst_points=matched_dst_points,
        match_indicator=match_indicator
    ) 

