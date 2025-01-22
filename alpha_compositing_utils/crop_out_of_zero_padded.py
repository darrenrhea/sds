import numpy as np


def crop_out_of_zero_padded(
    x: np.ndarray,
    i_min: int,
    i_max: int,
    j_min: int,
    j_max: int,
):
    """
    Sometimes you want to act as-if
    the image you are cropping out of is "infinitely large",
    i.e. 
    has row indices and column indices
    that can be any integer, not just those
    nonnegative integers that are within the bounds of the image,
    where the values outside the finite image are defined to be 0.

    This allows you to crop out of an image where the
    "cookie cutter" is aligned
    with the such that it falls partially (or completely) outside the image, the undefined part being zero by definition.
    """
    assert i_min < i_max
    assert j_min < j_max
    h = i_max - i_min
    w = j_max - j_min
    out_shape = (h, w) + x.shape[2:]
    # print(f"out_shape = {out_shape}")
    out = np.zeros(shape=out_shape, dtype=x.dtype)

    # print(f"When {i_min=} {i_max=} {j_min=} {j_max=}, the crop is {h=} tall and {w=} wide")
    # get the defined part of the crop.
    a = max(0, i_min)
    b = min(i_max, x.shape[0])
    c = max(0, j_min)
    d = min(j_max, x.shape[1])
    # print(f"The defined part of the crop is [{a}, {b}) x [{c}, {d})")
    if a < b and c < d:
        out[
            (a - i_min):(b-i_min),
            (c - j_min):(d-j_min),
            ...
        ] = x[a:b, c:d, ...]
    return out
    