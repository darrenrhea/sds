from get_polynomial_regression_coefficients import (
     get_polynomial_regression_coefficients
)

import numpy as np
from typing import Optional


def get_color_correction_polynomial_coefficients_from_from_to_mapping_array_f64(
    degree: int,
    from_to_mapping_array_f64: np.ndarray,
    subsample_this_many_points: Optional[int] = None,
) -> np.ndarray:
    """
    Hopefully you have already stacked a N x 2 x 3 np.array that says a bunch of
    datapoints for the regression, like a datapoint says
    I want this rgb to map to this rgb, where the range of the channels is [0.0, 1.0].
    """
    assert from_to_mapping_array_f64.shape[1] == 2
    assert from_to_mapping_array_f64.shape[2] == 3
    assert from_to_mapping_array_f64.dtype == np.float64

    # find a function that maps the each point in inputs to each point in the outputs
    # inputs = from_to_mapping_array_f64[:, 0, :]
    # outputs = from_to_mapping_array_f64[:, 1, :]

   
   
    if subsample_this_many_points is not None:
        row_indices = [x for x in range(from_to_mapping_array_f64.shape[0])]
        np.random.shuffle(row_indices)
        row_indices = row_indices[:subsample_this_many_points]
        row_indices = np.array(row_indices)
        smaller_f64 = from_to_mapping_array_f64[
            row_indices,
            :,
            :
        ].copy()
    else:
        smaller_f64 = from_to_mapping_array_f64

    
    coefficients = get_polynomial_regression_coefficients(
        from_to_mapping_array=smaller_f64,
        degree=degree
    )

    return coefficients


