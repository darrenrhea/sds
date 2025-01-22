

import numpy as np





def my_SO3_from_rodrigues(rodrigues):
    """

    SO3 = cos(angle) * I_3x3

    + 
    np.sin(angle) * [0  - uz   uy]
                    [uz    0  -ux]
                    [-uy  ux    0]

    + (1-cos) [ux ux  ux uy  ux uz]
              [uy ux  uy uy  uy uz]
              [uz ux  uz uy  uz uz]
    ]
    """
    angle = np.linalg.norm(rodrigues)
    # if not angle <= np.pi + 1e-12:
    #     print(f"Warning, angle is more than pi: {angle}")
    u = rodrigues / angle  # the unit length version is the rotation axis
    ux, uy, uz = u
    cosine = np.cos(angle)
    sine = np.sin(angle)
    I_3x3 = np.eye(3)
    anti = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0],])
    SO3 = cosine * I_3x3 + sine * anti + (1 - cosine) * np.outer(u, u)
    return SO3

