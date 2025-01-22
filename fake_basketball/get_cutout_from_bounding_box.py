from blur import blur
import numpy as np


def get_cutout_from_bounding_box(
    rgba_np_u8: np.ndarray,
    bbox: dict
) -> np.ndarray:
    """
    We will cut out a part of an importantpeople
    annotation as described by a bounding box,
    blurring the boundary a bit.
    TODO: can we get away with making it smaller than 3840x2160?
    Seems wasteful.
    """
    assert isinstance(bbox, dict)
    xmin = bbox["xmin"]
    xmax = bbox["xmax"]
    ymin = bbox["ymin"]
    ymax = bbox["ymax"]
    assert xmin < xmax
    assert ymin < ymax
    assert 0 <= xmin
    assert 0 <= ymin

    height = rgba_np_u8.shape[0]
    width = rgba_np_u8.shape[1]
    assert rgba_np_u8.shape[2] == 4
    cutout_rgba_np_uint8 = rgba_np_u8.copy()
    if xmin > 0:
        cutout_rgba_np_uint8[:, :xmin, 3] = 0
    if xmax < width - 1:
        cutout_rgba_np_uint8[:, xmax:, 3] = 0
    if ymin > 0:
        cutout_rgba_np_uint8[:ymin, :, 3] = 0
    if ymax < height - 1:
        cutout_rgba_np_uint8[ymax:, :, 3] = 0

    alpha = cutout_rgba_np_uint8[:, :, 3]
    blurred_alpha = blur(alpha, r=1.5)
    cutout_rgba_np_uint8[:, :, 3] = blurred_alpha

    #trim it:
    cutout_rgba_np_uint8 = cutout_rgba_np_uint8[ymin:ymax, xmin:xmax, :].copy()
    return cutout_rgba_np_uint8
