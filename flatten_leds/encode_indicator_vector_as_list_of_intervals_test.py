from encode_indicator_vector_as_list_of_intervals import (
     encode_indicator_vector_as_list_of_intervals
)
import numpy as np


def test_encode_indicator_vector_as_list_of_intervals_1():
    test_cases = [
        (
            [0, 0, 0, 0],
            [],
        ),
        (
            [1, 1, 1, 1],
            [(0, 4),],
        ),
        (
            [0, 0, 1, 1],
            [(2, 4),],
        ),
        (
            [1, 1, 0, 0],
            [(0, 2),],
        ),
        (
            [1, 0, 0, 1],
            [(0, 1), (3, 4)],
        ),
        (
            [1, 0, 1, 0],
            [(0, 1), (2, 3)],
        ),
        (
            [0, 1, 0, 1],
            [(1, 2), (3, 4)],
        ),
        (
            [0, 1, 1, 0],
            [(1, 3),],
        ),
    ]
   
    for x, y in test_cases:
        x = np.array(x).astype(bool)
        result = encode_indicator_vector_as_list_of_intervals(x)
        assert (
            result == y
        ), f"For input {x=}, the answer should be {y=}, but we got {result=} instead."


if __name__ == '__main__':
    test_encode_indicator_vector_as_list_of_intervals_1()
    print("encode_indicator_vector_as_list_of_intervals_test.py: all tests pass")