import numpy as np


def maybe_crop_to_nonzero_alpha_region(
    rgba_np_uint8: np.ndarray,
):
    """
    There are things that basically do this
    in bounding_boxes like
    find_bounding_box_for_true.py
    """

    alpha = rgba_np_uint8[:, :, 3]
    indicates_nonzero_rows = np.any(alpha, axis=1)
    # print(indicates_nonzero_rows)
    i_min = None
    for i in range(alpha.shape[0]):
        if indicates_nonzero_rows[i]:
            i_min = i
            break
    if i_min is None:
        return None
    
    for i in range(alpha.shape[0] - 1, -1, -1):
        if indicates_nonzero_rows[i]:
            i_max = i + 1
            break
    
    indicates_nonzero_cols = np.any(alpha, axis=0)
    # print(indicates_nonzero_cols)
    j_min = None
    for j in range(alpha.shape[1]):
        if indicates_nonzero_cols[j]:
            j_min = j
            break
    if j_min is None:
        return None
    for j in range(alpha.shape[1] - 1, -1, -1):
        if indicates_nonzero_cols[j]:
            j_max = j + 1
            break

    cropped = rgba_np_uint8[i_min:i_max, j_min:j_max, :]

    return cropped

