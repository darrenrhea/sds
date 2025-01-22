import numpy as np


def do_polynomial_fit(xs, ys, degree):
    coeffs = np.polyfit(xs, ys, deg=degree)

    model = np.poly1d(
        coeffs
    )
    return model
