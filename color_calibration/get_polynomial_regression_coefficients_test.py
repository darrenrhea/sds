from get_polynomial_regression_coefficients import (
     get_polynomial_regression_coefficients
)

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


# https://koaning.github.io/scikit-lego/user-guide/linear-models/
def test_get_polynomial_regression_coefficients_1():
    degree = 4
    num_data_points = 100
    input_vectors = np.random.randn(num_data_points, 3)
   

    poly = PolynomialFeatures(degree=degree)
    features = poly.fit_transform(input_vectors)
  
    coefficients = np.random.randn(features.shape[1], 3)
    output_vectors = np.dot(features, coefficients)
    
    
    from_to_mapping_array = np.zeros(
        shape=(
            num_data_points,
            2,
            3,
        ),
        dtype=np.float64
    )
    from_to_mapping_array[:, 0, :] = input_vectors
    from_to_mapping_array[:, 1, :] = output_vectors
    

    recovered = get_polynomial_regression_coefficients(
        from_to_mapping_array=from_to_mapping_array,
        degree=degree
    )
    print("coefficients should_be")
    print(coefficients)

    print("recovered coefficients are")
    print(recovered)
    assert np.allclose(coefficients, recovered, atol=1e-3)

   

if __name__ == "__main__":
    test_get_polynomial_regression_coefficients_1()