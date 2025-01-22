import numpy as np
from get_distorted_radius import (
     get_distorted_radius
)

def test_get_distorted_radius_1():
    k1 = -0.65511906
    k2 = -0.008709373
    
    distorted_radius = np.array(
        [
            0.03316598128732503,
            0.02874746463733838,
        ]
    )
    
    distorted_radius_squared = distorted_radius**2
    c = 1 + k1 * distorted_radius_squared + k2 * distorted_radius_squared**2
    undistorted_radius = c * distorted_radius
    print(f"{undistorted_radius=}")

    undistorted_radius_squared = undistorted_radius**2


    r_distorted = get_distorted_radius(
        undistorted_radius_squared=undistorted_radius_squared,
        k1=k1,
        k2=k2,
    )
   
    assert np.allclose(r_distorted, distorted_radius)
   

if __name__ == "__main__":
    test_get_distorted_radius_1()
    print("distort_radius_test.py has passed all tests.")