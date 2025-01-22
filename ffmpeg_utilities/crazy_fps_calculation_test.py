import numpy as np

from crazy_fps_calculation import (
     crazy_fps_calculation
)


def test_crazy_fps_calculation_1():
    input_output_pairs = [
        (1, .0125125000),
        (60, .9968291333),
        (57, .9467791333)
    ]

    for x, y in input_output_pairs:
        assert (
            crazy_fps_calculation(x) == y
        ), f"{crazy_fps_calculation(x)=} but should be {y}"

    print("Tests passed")
