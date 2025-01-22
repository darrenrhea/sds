import numpy as np


def encode_indicator_vector_as_list_of_intervals(x):
    """
    Given a one-dimensional discrete numpy array that
    contains only 0s and 1s (Falses and Trues),
    return a Python list of [start_inclusive, end_exclusive) pair-tuples
    that say where the 1 / True values are in the indicator vector, if anywhere.

    See the test
    encode_indicator_vector_as_list_of_intervals_test.py
    for input x, output y pairs.

    TODO: decide where this function should live.
    """
    assert isinstance(x, np.ndarray), "x must be a numpy array"
    assert x.ndim == 1, "Only 1D arrays make sense to be passed to encode_indicator_vector_as_list_of_intervals"
    assert np.max(x) <= 1, "Only 0s and 1s are allowed in x passed to encode_indicator_vector_as_list_of_intervals"
    assert np.min(x) >= 0, "Only 0s and 1s are allowed in x passed to encode_indicator_vector_as_list_of_intervals"
    assert (
        x.dtype == bool
        or x.dtype == np.uint8
        or x.dtype == np.int8
        or x.dtype == np.int16
        or x.dtype == np.int32
        or x.dtype == np.int64
    ), "Only boolean or integer types are allowed in x passed to encode_indicator_vector_as_list_of_intervals"
    
    
    # prepend a zero prior to the vector x so that a 1 in the 0-ith entry of x position causes an upswing in diff.
    # append a zero after the vector so that a 1 in the last entry of x causes a downswing in diff.
    diff = np.diff(x.astype(np.int8), prepend=0, append=0)
    # find the indices where the value changes to 1
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0]
    assert len(start_indices) == len(end_indices), "ERROR: This should never happen?!"
    return list(zip(start_indices, end_indices))



