def radial_polynomial_map(k1, k2, k3, p1, p2, x, y):
    """
    In the Nuke Lens model, this is used to go from
    distorted to undistorted.

    a = x' = x + x * (k1 * r^2 + k2 * r^4 + k3 * r^6) + p1 * (r^2 + 2*x^2) + 2 * p2 * x * y
    b = y' = y + y * (k1 * r^2 + k2 * r^4 + k3 * r^6) + p2 * (r^2 + 2*y^2) + 2 * p1 * x * y 
    """
    r2 = x ** 2 + y ** 2
    c = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    a = c * x + p1 * (r2 + 2 * x**2) + 2 * p2 * x * y
    b = c * y + p2 * (r2 + 2 * y**2) + 2 * p1 * x * y 
    return (a, b)