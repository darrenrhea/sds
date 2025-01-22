"""
Tests for nuke_lens_distortion.py
"""

from functools import partialmethod
import sys
import time
import numpy as np
from nuke_lens_distortion import (
    radial_polynomial_map,
    inverse_of_radial_polynomial_map,
    nuke_world_to_pixel_coordinates,
    get_floor_point_world_coordinates_of_pixel_coordinates
)

from CameraParameters import CameraParameters

"""
k1 = -0.21464084305343936, k2 = -0.0010012818559894483
orig r2 = r^2
new r2 = (c*x)^2 + (c*y)^2 = c^2 * r2 = (1 + k1 r2 + k2 *r2^2) * r2
r2 |-> (1 + k1*r2 + k2 *r2^2)^2 * r2 should be monotonic on [0, 1]
(- 2k1 +- np.sqrt(4*k1**2 - 4 * 3 * k2 * 1)) / 6 k2
"""


def test_inverse_of_radial_polynomial_map():
    """
    We fixate k1, k2, k3 and see that the inverse map successfully inverts the forward map.
    """
    num_attempts = 10000
    for _ in range(num_attempts):
        k1 = 0.03 * np.random.randn()
        k2 = 0.003 * np.random.randn()
        k3 = 0.0001 * np.random.randn()
        p1 = 0.001 * np.random.randn()
        p2 = 0.001 * np.random.randn()
        actual_x = 2.2 * np.random.rand() - 1.1  # in the interval (-1.1, 1.1)
        actual_y = 2.2 * np.random.rand() - 1.1  # in the interval (-1.1, 1.1)
        actual_r2 = actual_x ** 2 + actual_y ** 2
        a, b = radial_polynomial_map(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, x=actual_x, y=actual_y)
        recovered_x, recovered_y = inverse_of_radial_polynomial_map(
            k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, a=a, b=b
        )
        recovered_r2 = recovered_x ** 2 + recovered_y ** 2
        if not (
            np.abs(actual_x - recovered_x) < 1e-10
            and np.abs(actual_y - recovered_y) < 1e-10
        ):
            print(f"failed: k1 = {k1}, k2 = {k2}, k3={k3}")
            print(f"actual_x = {actual_x}, actual_y = {actual_y}")
            print(f"actual_r2 = {actual_r2}")
            print(
                f"distorted_r2 = {(1 + k1 * actual_r2 + k2 * actual_r2**2 + k3 * actual_r2**3)**2 * actual_r2}"
            )
            print(f"r2 = {actual_r2}")
            print(f"a = {a}, b = {b}")
            print(f"recovered_x = {recovered_x}, recovered_y = {recovered_y}")
            print(
                f"distorted_r2 = {(1 + k1 * recovered_r2 + k2 * recovered_r2**2  + k3 * recovered_r2**3)**2 * recovered_r2}"
            )

            print(
                radial_polynomial_map(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, x=recovered_x, y=recovered_y)
            )
            assert False

    print(f"test_inverse_of_radial_polynomial_map passed {num_attempts} attempts")


if __name__ == "__main__":
    test_inverse_of_radial_polynomial_map()
    