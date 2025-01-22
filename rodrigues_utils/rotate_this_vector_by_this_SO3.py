

import numpy as np





def rotate_this_vector_by_this_SO3(vec, rodrigues):
    """
    Returns the result of rotating vector vec by the rotation defined by the angle-axis vector rodrigues.
    The Rodrigues vector's length is the angle (in radians) to rotate in right-hand rule about the axis.
    """
    angle = np.linalg.norm(rodrigues)
    if not angle <= np.pi + 1e-12:
        print(f"Warning, angle is more than pi: {angle}")
    u = rodrigues / angle  # the unit length version is the rotation axis

    term1 = vec * np.cos(angle)
    term2 = (np.cross(u, vec)) * np.sin(angle)
    term3 = u * (u.dot(vec)) * (1 - np.cos(angle))
    return term1 + term2 + term3

