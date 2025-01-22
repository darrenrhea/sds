import numpy as np
import better_json as bj
from pathlib import Path

from math import comb


def save_color_correction_as_json(
    degree: int,
    coefficients: np.ndarray,
    out_path: Path
) -> None:
    assert isinstance(out_path, Path)
    assert isinstance(degree, int)
    assert isinstance(coefficients, np.ndarray)
    # assert coefficients.ndim == 2
    assert coefficients.shape[1] == 3
    num_coefficients_should_be = comb(degree + 3, 3)
    assert (
        coefficients.shape[0] == num_coefficients_should_be
    ), f"ERROR: {coefficients.shape=}, {num_coefficients_should_be=}" 

    coefficients_as_list_of_lists = coefficients.tolist()
    for i in range(num_coefficients_should_be):
        for j in range(3):
            assert isinstance(coefficients_as_list_of_lists[i][j], float)

    obj = dict(
        degree=degree,
        coefficients=coefficients_as_list_of_lists,    
    )

    bj.dump(
        obj=obj,
        fp=out_path,
    )

