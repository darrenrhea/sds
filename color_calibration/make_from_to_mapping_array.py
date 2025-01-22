import numpy as np


def make_from_to_mapping_array(color_names, color_map):
    """
    We often want to express that these points if R^3
    should be transformed to these other points in R^3,
    say in the context of color correction.
    """
    mapping = np.zeros(
        shape=(
            len(color_names),
            2,
            3,
        ),
        dtype=np.float64
    )

    for i, color_name in enumerate(color_names):
        mapping[i, 0, :] = color_map[color_name]["is"]
        mapping[i, 1, :] = color_map[color_name]["should_be"]
    
    return mapping