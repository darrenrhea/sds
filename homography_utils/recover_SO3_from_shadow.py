import numpy as np
from numpy import core

def SO3ify(R):
    Q, _ = np.linalg.qr(R)
    for j in range(3):
        if np.dot(Q[:,j], R[:,j]) < 0:
            Q[:, j] *= -1
    return Q

def make_random_SO3_matrix(theta_degrees, phi_degrees, gamma_degrees):
    theta = theta_degrees / 180 * np.pi  # say 15 degrees rotation
    phi = phi_degrees / 180 * np.pi  # say 7 degrees rotation
    gamma = gamma_degrees / 180 * np.pi  # say 7 degrees rotation

    R = (
        np.array(
            [
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)]
            ]
        )
        @
        np.array(
            [
                [np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1],
            ]
        )
        @
        np.array(
            [
                [np.cos(gamma), 0, -np.sin(gamma)],
                [0, 1, 0],
                [np.sin(gamma), 0, np.cos(gamma)]
            ]
        )
    )
    
    assert np.allclose(
        np.eye(3),
        R @ R.T
    )

    return R

def make_shadow(R):
    """
    We define the "shadow" of an SO3 matrix to be what you get if you
    take its 3 row vectors and project them into the x-y plane (drop the z coordinate)
    and then lose length information about the 3 projections by dividing each
    by its own random nonnegative scalar.
    Thing of the original SO3 matrix as being 3 half-infinite orthogonal rays leaving the origin
    (each ray is all positive multiples a row of R)
    and then you can project the rays into the x-y plane to get 3 half-infinite rays in the x-y plane.
    Can you, from just knowning those 3 rays in the xy-plane, get back the original SO3 matrix?
    The answer is yes.
    """
    determinant = np.linalg.det(R)
    assert abs(determinant - 1.0) < 1e-8, f"The determinant is {determinant}"
    shadow = R[:, :2].copy()
    a = np.random.rand() + 0.1
    b = np.random.rand() + 0.1
    c = np.random.rand() + 0.1
    print(f"a b c should be {a} {b} {c}")
    shadow[0, :] /= a
    shadow[1, :] /= b
    shadow[2, :] /= c

    should_be_one = a**2 * shadow[0, 0]**2 + b**2 * shadow[1, 0]**2 + c**2 * shadow[2, 0]**2
    should_also_be_one = a**2 * shadow[0, 1]**2 + b**2 * shadow[1, 1]**2 + c**2 * shadow[2, 1]**2
    should_be_zero = a**2 * shadow[0, 0] * shadow[0, 1] + b**2 * shadow[1, 0] * shadow[1, 1] + c**2 * shadow[2,0]*shadow[2, 1]
    print(f"should_be_one = {should_be_one}")
    print(f"should_also_be_one = {should_be_one}")
    print(f"should_be_zero = {should_be_zero}")
    return shadow


def recover_SO3_from_its_shadow(shadow):
    """   
    The thought is that a^2, b^2, c^2 solve this system of affine equations:
    should_be_one = a^2 * shadow[0, 0]^2 + b^2 * shadow[1, 0]^2 + c^2 * shadow[2, 0]^2
    should_also_be_one = a^2 * shadow[0, 1]^2 + b^2 * shadow[1, 1]^2 + c^2 * shadow[2, 1]^2
    should_be_zero = a^2 * shadow[0, 0] * shadow[0, 1] + b^2 * shadow[1, 0] * shadow[1, 1] + c^2 * shadow[2,0]*shadow[2, 1]
    """

    M = np.array(
        [
            [shadow[0, 0]**2,             shadow[1, 0]**2,             shadow[2, 0]**2],
            [shadow[0, 1]**2,             shadow[1, 1]**2,             shadow[2, 1]**2],
            [shadow[0, 0] * shadow[0, 1], shadow[1, 0] * shadow[1, 1], shadow[2,0]*shadow[2, 1]],
        ]
    )
    print(f"M={M}")
    det_M = np.linalg.det(M)
    s00, s01 = shadow[0]
    s10, s11 = shadow[1]
    s20, s21 = shadow[2]

    """
    alt_det_M = (
        s00**2 * (s11**2*s20*s21 - s21**2*s10*s11)
        -
        s10**2 * (s01**2*s20*s21-s21**2*s00*s01)
        +
        s20**2 * (s01**2*s10*s11 - s11**2*s00*s01)
    )
    assert abs(det_M - alt_det_M) < 1e-8, f"{abs(det_M - alt_det_M)}"
    """

    print(f"det(M) = {det_M}")
    v = np.array([1, 1, 0])
    abc_squared = np.linalg.solve(M, v)
    print(abc_squared)
    if np.any(abc_squared < 0):
        return False, None, None, None, None
    abc = abc_squared**0.5
    a, b, c = abc
    recovered_R = np.zeros(shape=(3, 3))
    recovered_R[0, :2] = a * shadow[0, :]
    recovered_R[1, :2] = b * shadow[1, :]
    recovered_R[2, :2] = c * shadow[2, :]
    cross_product = np.cross(recovered_R[:,0], recovered_R[:,1])
    recovered_R[:,2] = cross_product
    return True, recovered_R, a, b, c



def recover_SO3_from_its_shadow_aa(shadow):
    """   
    The thought is that a^2 and c^2 solve this system of affine equations:
    should_be_one = a^2 * shadow[0, 0]^2 + a^2 * shadow[1, 0]^2 + c^2 * shadow[2, 0]^2
    should_also_be_one = a^2 * shadow[0, 1]^2 + a^2 * shadow[1, 1]^2 + c^2 * shadow[2, 1]^2
    should_be_zero = a^2 * shadow[0, 0] * shadow[0, 1] + a^2 * shadow[1, 0] * shadow[1, 1] + c^2 * shadow[2,0]*shadow[2, 1]
    """

    M = np.array(
        [
            [shadow[0, 0]**2 + shadow[1, 0]**2,             shadow[2, 0]**2],
            [shadow[0, 1]**2 + shadow[1, 1]**2,             shadow[2, 1]**2],
            [shadow[0, 0] * shadow[0, 1] + shadow[1, 0] * shadow[1, 1], shadow[2,0]*shadow[2, 1]],
        ]
    )
    
    s00, s01 = shadow[0]
    s10, s11 = shadow[1]
    s20, s21 = shadow[2]

  

    v = np.array([1, 1, 0])
    ac_squared, residuals, _, _= np.linalg.lstsq(M, v, rcond=None)
    print(ac_squared)
    if np.any(ac_squared < 0):
        return False, None, None, None, None
    ac = ac_squared**0.5
    a, c = ac
    recovered_R = np.zeros(shape=(3, 3))
    recovered_R[0, :2] = a * shadow[0, :]
    recovered_R[1, :2] = a * shadow[1, :]
    recovered_R[2, :2] = c * shadow[2, :]
    cross_product = np.cross(recovered_R[:,0], recovered_R[:,1])
    recovered_R[:,2] = cross_product
    recovered_R = SO3ify(recovered_R)
    return True, recovered_R, a, a, c



def test_recover_SO3_from_its_shadow():
    R = make_random_SO3_matrix(
        np.random.randint(359),
        np.random.randint(359),
        np.random.randint(359)
    )
    print(f"\n\n\n\noriginal R=\n{R}")
    shadow = make_shadow(R)
    print(f"shadow={shadow}")
    success, recovered_R, a, b, c = recover_SO3_from_its_shadow(shadow)
    assert success
    print(f"recovered_R=\n{recovered_R}")
    assert np.allclose(R, recovered_R)
    

def test_particular_R():
    R = np.array(
        [
            [ 0.73624898, -0.51552709, -0.43837115],
            [ 0.57357644,  0.81915204,  0.        ],
            [ 0.35909262, -0.25143936,  0.89879405]
        ]
    )
    print(np.linalg.det(R))
    print(f"R=\n{R}")
    R = SO3ify(R)
    print("replaced by")
    print(R)
    print(R @ R.T)
    assert np.allclose(R @ R.T, np.eye(3))
    shadow = make_shadow(R)
    success, recovered_R, a, b, c = recover_SO3_from_its_shadow(shadow)
    assert success
    print(f"The recovered_R is\n{recovered_R}")
    print(np.linalg.norm(R-recovered_R, ord='fro'))
    print(
        np.sum(
            (R-recovered_R)**2
        )**0.5
    )

if __name__ == "__main__":
    test_particular_R()
    for cntr in range(10):
        print(f"trial {cntr}")
        test_recover_SO3_from_its_shadow()



   

    