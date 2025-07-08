from rotate_this_vector_by_this_rodrigues import (
     rotate_this_vector_by_this_rodrigues
)

import numpy as np





def SO3_from_rodrigues(rodrigues):
    """
    HOTSPOT: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    """
    # r = R.from_rotvec(rodrigues)
    # return r.as_matrix()
    rotation_matrix = np.eye(3, dtype=np.float64)
    if np.linalg.norm(rodrigues) < 1e-12:
        return rotation_matrix
    rotation_matrix[:, 0] = rotate_this_vector_by_this_rodrigues(
        vec=np.array([1, 0, 0]), rodrigues=rodrigues
    )
    rotation_matrix[:, 1] = rotate_this_vector_by_this_rodrigues(
        vec=np.array([0, 1, 0]), rodrigues=rodrigues
    )
    rotation_matrix[:, 2] = rotate_this_vector_by_this_rodrigues(
        vec=np.array([0, 0, 1]), rodrigues=rodrigues
    )
    return rotation_matrix

