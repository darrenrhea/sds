import numpy as np


def calculate_metrics(
    from_to_mapping_array_f64: np.ndarray
) -> float:
    """
    This function is for calculating the metrics of the alignment quality.
    """
    a = from_to_mapping_array_f64[:, 0, :]
    b = from_to_mapping_array_f64[:, 1, :]
    a_centered = a - np.mean(a, axis=0)
    b_centered = b - np.mean(b, axis=0)
    a_normalized = a_centered / np.linalg.norm(a_centered, axis=0)
    b_normalized = b_centered / np.linalg.norm(b_centered, axis=0)
    for c in range(3):
        assert np.allclose(np.linalg.norm(a_normalized[:, c]), 1)
        assert np.allclose(np.linalg.norm(b_normalized[:, c]), 1)
    corr = np.sum(a_normalized * b_normalized) / 3.0
    return corr


