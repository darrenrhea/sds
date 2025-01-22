from load_color_correction_from_json import (
     load_color_correction_from_json
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import numpy as np
from typing import Tuple


def load_color_correction_from_sha256(
    sha256: str,
) -> Tuple[int, np.ndarray]:
    """
    This function will load the color correction from its sha256.
    """
    color_correction_json_path = get_file_path_of_sha256(
        sha256=sha256
    )

    return load_color_correction_from_json(
        json_path=color_correction_json_path
    )
    