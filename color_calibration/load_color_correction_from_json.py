import numpy as np
import better_json as bj
from pathlib import Path
from math import comb
from typing import Tuple


def load_color_correction_from_json(
    json_path: Path
) -> Tuple[int, np.ndarray]:
    """
    This function will load the color correction from a json file.
    """
    assert isinstance(json_path, Path)
    assert json_path.resolve().is_file()

    obj = bj.load(json_path)
    degree = obj["degree"]
    assert isinstance(degree, int)
    assert degree >= 1

    coefficients_as_list_of_lists = obj["coefficients"]
    assert isinstance(coefficients_as_list_of_lists, list)
    assert len(coefficients_as_list_of_lists) > 0
    for x in coefficients_as_list_of_lists:
        assert isinstance(x, list)
        assert len(x) == 3
        for y in x:
            assert isinstance(y, float)

    coefficients = np.array(coefficients_as_list_of_lists)
    assert coefficients.shape[1] == 3
    num_coefficients_should_be = comb(degree + 3, 3)
    assert (
        coefficients.shape[0] == num_coefficients_should_be
    ), f"ERROR: {coefficients.shape=}, {num_coefficients_should_be=}"
    return degree, coefficients

