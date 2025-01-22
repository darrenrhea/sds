import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from sklego.linear_model import LADRegression
# pip install scikit-lego

def get_polynomial_regression_coefficients(
    from_to_mapping_array: np.ndarray,
    degree: int
):
    """
    Trying to future proof this function by not assuming 8-bit color depth.
    
    Given n x 2 x 3 np.ndarray of floats in [0.0, 1.0]
    like 
    [
        [
            [r0, g0, b0],
            [r1, g1, b1]
        ],
        [
            [r2, g2, b2],
            [r3, g3, b3]
        ],
        ...
    fit a regression model to predict the second color from the first color.
    """
    X = from_to_mapping_array[:, 0, :]
    assert np.max(X) <= 1.01, f"This is intended to be a float in [0.0, 1.0] but max is {np.max(X)}"
    assert np.min(X) >= -0.01, f"This is intended to be a float in [0.0, 1.0] but min is {np.min(X)}"
    poly = PolynomialFeatures(degree=degree)
    features = poly.fit_transform(X)

    coefficients = np.zeros(
        (features.shape[1], 3)
    )

    for c in range(3):
        y = from_to_mapping_array[:, 1, c]
        # https://koaning.github.io/scikit-lego/api/linear-model/#sklego.linear_model.LADRegression
        model = LADRegression(
            fit_intercept=False,
        )
        model.fit(features, y)
        coefficients[:, c] = model.coef_

    return coefficients

    
    