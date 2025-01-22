import numpy as np


def do_quadratic_fit(xs, ys):
    coeffs = np.polyfit(xs, ys, deg=2)

    model = np.poly1d(
        coeffs
    )
    return model
