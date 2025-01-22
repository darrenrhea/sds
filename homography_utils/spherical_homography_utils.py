import numpy as np
from homography_utils import map_points_through_homography




def form_H(s, f0, f1, R):
    # print(R)
    # print(np.diag([1, 1, 1 / f1]) @ R @ np.diag([1, 1, f0]))

    H = s * np.diag([1, 1, 1 / f1]) @ R @ np.diag([1, 1, f0])
    return H


def solve_for_R_s_f0_and_f1(H):
    """
    Assuming that H is exactly a spherical homography,
    what are the focal lengths f0, f1, the scale s, and the SO3 rotation matrix R
    such that
    H = s * diag(1, 1, 1/f1) * R * diag(1, 1, f0) ?  
    """
    print(f"We are going to solve for R, s, f0, and f1 given the H {H}")
    (H00, H01, H02, H10, H11, H12, H20, H21, H22) = H.flatten()
    # Find the f0 such that 
    # (H00, H01, H0 / f0) \cdot (H10, H11, H12/f0) = 0
    print(f"{H00 * H10 + H01 * H11=}")
    print(f"{H02 * H12=}")
    # 0 = H00 * H10 + H01 * H11 + H02 * H12 / f0**2
    f0 = np.sqrt(
        - (H02 * H12) / (H00 * H10 + H01 * H11)
    )
    print(f"We claim {f0=}")

    # Find the f1 such that 
    # (H00, H10, H20 * f1) \cdot (H01, H11, H21 * f1) = 0
    # 0 = H00 * H01 + H10 * H11 + H20 * H21 * f1**2
    f1 = np.sqrt(
        - (H00 * H01 + H10 * H11) / (H20 * H21)
    )
    print(f"We claim {f1=}")
    R = H.copy()
    R[:, 2] /= f0
    R[2, :] *= f1
    s = np.sqrt(np.sum(R**2) / 3)
    print(f"We claim {s=}")
    print(f"We claim {R=}")
    recovered_H = form_H(s=s, f0=f0, f1=f1, R=R)
    print(f"We claim {recovered_H=}")
    distance = np.linalg.norm(H - recovered_H)
    print(f"We claim {distance=}")
    return R, s, f0, f1
