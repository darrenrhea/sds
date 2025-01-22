import numpy as np


def get_from_to_mapping_array(
    from_rgb_np: np.ndarray,
    to_rgb_np: np.ndarray,
    color_sampling_mask_hw_np_u8: np.ndarray,
) -> np.ndarray:
    """
    Given two images and a mask, all of the same height and width,
    this function returns a 3D array of shape (N, 2, 3) where N is the number of pixels
    in the mask that are on, which currently means 255.

    We ask that the two images are floating point and the mask is uint8, where 255 means sample,
    and 0 means don't sample.
    """
    assert from_rgb_np.shape == to_rgb_np.shape
    assert from_rgb_np.shape[:2] == color_sampling_mask_hw_np_u8.shape

    ijs = np.argwhere(color_sampling_mask_hw_np_u8 > 128)

    inputs = from_rgb_np[ijs[:, 0], ijs[:, 1], :]
    outputs = to_rgb_np[ijs[:, 0], ijs[:, 1], :]
    
    # form the regression data:
    from_to_mapping_array_f64 = np.zeros(
        shape=(
            ijs.shape[0],
            2,
            3,
        ),
        dtype=np.float64
    )
    from_to_mapping_array_f64[:, 0, :] = inputs
    from_to_mapping_array_f64[:, 1, :] = outputs
    
    assert from_to_mapping_array_f64.shape[0] == ijs.shape[0]
    assert from_to_mapping_array_f64.shape[1] == 2
    assert from_to_mapping_array_f64.shape[2] == 3
    assert from_to_mapping_array_f64.dtype == np.float64

    return from_to_mapping_array_f64

