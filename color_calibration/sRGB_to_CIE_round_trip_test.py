from CIE_linear_XYZ_to_sRGB import (
     CIE_linear_XYZ_to_sRGB
)
from sRGB_to_CIE_linear_XYZ import (
     sRGB_to_CIE_linear_XYZ
)

import numpy as np


def test_sRGB_to_CIE_round_trip():
    #rgb_values = np.meshgrid(np.linspace
    r, g, b = np.meshgrid(
        np.linspace(0, 255, 256),
        np.linspace(0, 255, 256),
        np.linspace(0, 255, 256),
    )
    rgb_values = np.hstack(
        (r.ravel(), g.ravel(), b.ravel()),
    )
    print(f"{rgb_values.shape=}")


    rgb_values = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [0, 0, 0]
        ]
    )
    xyz_values = sRGB_to_CIE_linear_XYZ(rgb_values)
    rgb_values_again = CIE_linear_XYZ_to_sRGB(xyz_values)

    residuals = np.abs(rgb_values - rgb_values_again)
    L_1_error = np.mean(residuals)
    L_2_error = np.sqrt(np.mean(residuals**2))
    L_infinity_error = np.max(residuals)
    print(f"{L_1_error=}")
    print(f"{L_2_error=}")
    print(f"{L_infinity_error=}")

    assert np.all(np.abs(rgb_values - rgb_values_again) < 1e-10)

if __name__ == "__main__":
    test_sRGB_to_CIE_round_trip()
    print("sRGB_to_CIE_round_trip_test.py: all tests pass yo")