
import numpy as np
from SO3_from_rodrigues import (
     SO3_from_rodrigues
)




def canonicalize_rodrigues(bad_rodrigues):
    """
    Sometimes solvers, e.g. Ceres, will return a Rodrigues angle-axis vector
    with a norm greater than pi.
    This makes an equivalent angle-axis vector with norm between 0 and pi.
    """
    original_length = np.linalg.norm(bad_rodrigues)
    u = bad_rodrigues / original_length
    how_many_two_pis = np.floor(original_length / (2 * np.pi))
    new_length = original_length - 2 * np.pi * how_many_two_pis
    if new_length > np.pi:
        u = -u
        new_length = 2 * np.pi - new_length
    assert new_length < np.pi + 1e-12
    assert 0 < new_length
    canonical = new_length * u
    assert np.allclose(SO3_from_rodrigues(bad_rodrigues), SO3_from_rodrigues(canonical))
    return canonical

