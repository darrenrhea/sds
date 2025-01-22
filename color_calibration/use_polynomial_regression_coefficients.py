import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def use_polynomial_regression_coefficients(
    degree: int,
    coefficients: np.ndarray,
    input_vectors: np.ndarray
) -> np.ndarray:
    """
    If you have coefficients from a polynomial regression model,
    and you have input vectors, you can use this function to predict.
    """
    poly = PolynomialFeatures(degree=degree)
    features = poly.fit_transform(input_vectors)  
    output_vectors = np.dot(features, coefficients)
    return output_vectors
    