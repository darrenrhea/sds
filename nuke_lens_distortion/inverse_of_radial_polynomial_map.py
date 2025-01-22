import numpy as np


def inverse_of_radial_polynomial_map(k1, k2, k3,  p1, p2, a, b):
    """
    For a person using the Nuke lens distortion model,
    you will need this if you want to convert an undistorted location to
    a distorted location, say for drawing wireframes onto distorted images.

    Suppose that k1, k2, k3 in R are fixed real numbers.
    Suppose that the function f: R^2 --> R^2
    if defined as f(x, y) = [ f_1(x,y) f_2(x, y)]
    (1 + k1 * r^2 + k2 * r^4  + k3 * r^6) * (x, y)
    where r^2 := x^2 + y^2.
    Then this is the inverse function g: R^2 --> R^2.
    i.e. if (x, y) = g(a, b) then f(x, y) = (a, b).

    NOTE: there is a good test for this: so test via typing "pytest" if you screw with it.
    """
    x, y = a, b # decent guess since the distortion is smallish
    max_iterations = 10
    iterations = 0
    while True:
        # h = 1e-7
        # a0, b0 = radial_polynomial_map(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, x=x, y=y)
        # a_dx, b_dx = radial_polynomial_map(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, x=x + h, y=y)
        # a_dy, b_dy = radial_polynomial_map(k1=k1, k2=k2, k3=k3, p1=p1, p2=p2, x=x, y=y+h)
        # print(f"numerical partial of f_1 wrt x is {(a_dx - a0) / h}")
        # print(f"numerical partial of f_1 wrt y is {(a_dy - a0) / h}")
        # print(f"numerical partial of f_2 wrt x is {(b_dx - b0) / h}")
        # print(f"numerical partial of f_2 wrt y is {(b_dy - b0) / h}")
        
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        r4 = r2 * r2
        r6 = r4 * r2
        c = 1 + k1 * r2 + k2 * r4 + k3 * r6
        f1 = c * x + p1 * (r2 + 2 * x2) + 2 * p2 * xy
        f2 = c * y + p2 * (r2 + 2 * y2) + 2 * p1 * xy 
        residual_1 = f1 - a
        residual_2 = f2 - b
        # residual_vector = np.array([residual_1, residual_2])
        if np.abs(residual_1) < 1e-12 and np.abs(residual_2) < 1e-12:
            break

        temp = 2 * (k1 + 2 * k2 * r2 + 3 * k3 * r4)  # this subexpression happens a lot
        partial_f1_wrt_x = temp * x2 + c + (p1 * 6*x + 2 * p2 * y)
        partial_f1_wrt_y = temp * xy + (p1 * 6*x + p2*y) + (p1 * 2*y + 2 * p2 * x)

        partial_f2_wrt_x = temp * xy + (p2 * 2*x + 2 * p1 * y)
        partial_f2_wrt_y = temp * y2 + c + (p2 * 6*y + 2 * p1 * x)

        # print(f"symbolic partial of f_1 wrt x is {partial_f1_wrt_x}")
        # print(f"symbolic partial of f_1 wrt y is {partial_f2_wrt_x}")
        # print(f"symbolic partial of f_2 wrt x is {partial_f1_wrt_y}")
        # print(f"symbolic partial of f_2 wrt y is {partial_f2_wrt_y}")

        # we save 0.1 seconds per image by not instantiating J
        # J = np.array(
        #     [[partial_f1_wrt_x, partial_f1_wrt_y], [partial_f2_wrt_x, partial_f2_wrt_y]]
        # )
        # assert np.linalg.det(J) != 0  # this can cause 0.5 seconds per image!

        # save about 0.8 seconds per image by not doing this:
        # J_inverse = np.linalg.inv(J)
        # delta = np.dot(J_inverse, residual_vector)
        
        # but instead this:
        #delta_x, delta_y = np.linalg.solve(J, residual_vector) 

        # but this is 0.5 seconds even faster:
        denom = partial_f1_wrt_x * partial_f2_wrt_y - partial_f1_wrt_y * partial_f2_wrt_x
        delta_x = (residual_1 * partial_f2_wrt_y - partial_f1_wrt_y * residual_2) / denom
        delta_y = (partial_f1_wrt_x * residual_2 - residual_1 * partial_f2_wrt_x) / denom
        """
        [A B] dx   [res1]
        [C D] dy = [res2]  
       
        dx = (res1 * D - B*res2) / (AD-BC)
        dy = (A*res2 - res1* C) / (AD-BC)
        """

        x -= delta_x
        y -= delta_y
        iterations += 1
        if iterations >= max_iterations:
            break

    if iterations == max_iterations:
        print(f"Strangely we got to max_iterations={max_iterations}")
        print(f"k1={k1} k2={k2} k3={k3} p1={p1} p2={p2} a={a} b={b}")

    return (x, y)
