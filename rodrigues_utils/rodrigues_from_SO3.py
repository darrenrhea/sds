from SO3_from_rodrigues import (
     SO3_from_rodrigues
)


import numpy as np





def rodrigues_from_SO3(so3):
    """
    Given a 3x3 rotation matrix, what Rodrigues vector is equivalent?
    """
    eigenvalues, eigenvectors = np.linalg.eig(so3)

    best_so_far = 2.0
    for j in range(3):
        dist = np.abs(eigenvalues[j] - 1.0)
        if dist < best_so_far:
            best_so_far = dist
            best_j = j

    u = np.real(
        eigenvectors[:, best_j]
    )  # u is the (necessarily real) unit eigenvector of eigenvalue 1
    complex_j = (best_j + 1) % 3  # the column index of a different eigenvector
    cosine = np.real(eigenvalues[complex_j])
    angle = np.arccos(cosine)
    candidate = angle * u
    if np.linalg.norm(SO3_from_rodrigues(candidate) - so3) > 0.000001:
        u = -u
        candidate = angle * u
    assert np.linalg.norm(SO3_from_rodrigues(candidate) - so3) < 0.00001

    return candidate

