import numpy as np


def get_distorted_radius(
    undistorted_radius_squared,
    k1,
    k2
):
    # These u, va need to be distorted:
    # u_undistorted = 0.123
    # v_undistorted = 0.14

    # r_undistorted2 = u_undistorted**2 + v_undistorted**2

    # The undistortion happened to the radius only via:
    # rdistorted2 = distorted_us**2 + distorted_vs**2
    # c = 1 + k1 * rdistorted2 + k2 * rdistorted2**2
    # r_undistored = rdistorted * c
    # so we need to solve for rdistorted:
    # call r_distorted**2 "x".
    # r_undistorted2 = u**2 + v**2
    # r_undistorted2 = (1 + k1 * rdist**2 + k2 * rdist**4)^2 * rdist2
    # r_undistorted2 = (1 + k1 * x + k2 * x^2)^2 * x
    # NOTE:
    #  (1 + k1 * x + k2 * x**2)**2 * x =
   

    # k1**2*x**3 + 2*k1*k2*x**4 + 2*k1*x**2 + k2**2*x**5 + 2*k2*x**3 + x
    # r_undistorted2 = k1**2*x**3 + 2*k1*k2*x**4 + 2*k1*x**2 + k2**2*x**5 + 2*k2*x**3 + x
    # 0 = - r_undistorted2 + x + (2 * k1) * x**2 + (2*k2) * x**3 + (k1**2*) x**3 + 2*k1*k2*x**4 + k2**2*x**5


    # BEGIN calculate world velocities of the backwards light rays:
    # unpack the rows into the camera's axes:

    # the initial guess for the distorted radius is the undistorted radius:
    original_undistorted_radius_squared = undistorted_radius_squared.copy()

    ignore_this = undistorted_radius_squared > ((1)**2 + 9/16**2) * 1.2 ** 2
    x = undistorted_radius_squared

    # iterate Newton's method:
    # new_x = x - f(x) / f'(x)
    for _ in range(3):
        numerator = (
            - undistorted_radius_squared
            +
            x
            +
            (2 * k1) * x**2
            +
            (2 * k2 + k1**2) * x**3
            +
            2 * k1 * k2 * x**4
            +
            k2**2 * x**5
        )

        denominator = (
            1
            +
            2 * (2 * k1) * x
            +
            3 * (2 * k2 + k1**2) * x**2
            +
            4 * 2 * k1 * k2 * x**3
            +
            5 * k2**2 * x**4
        
        )
        x = x - numerator / denominator
    
    # if np.any(x < 0):
    #     print(f"Warning: x = {x[x < 0]} < 0")
    x[ignore_this] = original_undistorted_radius_squared[ignore_this]
    x = np.maximum(x, 0)
    return np.sqrt(x)
