import cv2
from tqdm import tqdm
import PIL.Image
from colorama import Fore, Style
from typing import List
from pathlib import Path
import numpy as np


def write_frame(
    frame: np.ndarray,  # may actually be grayscale and thus ndim = 2
    output_file_path: Path,
    original_height: int,
    original_width: int
) -> None:
    assert frame.ndim == 2, f"{frame.ndim=}"
    # tqdm.write(f'writing {output_file_path}')
    full_size = cv2.resize(frame, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    PIL.Image.fromarray(full_size).save(output_file_path, format="PNG")
