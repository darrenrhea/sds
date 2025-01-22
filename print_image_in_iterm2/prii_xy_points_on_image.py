from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from typing import List, Optional, Tuple
import PIL.Image
from pathlib import Path
import numpy as np


def prii_xy_points_on_image(
    xys: np.ndarray | List[Tuple[int, int]],
    image: str | Path | np.ndarray | PIL.Image.Image,
    output_image_file_path: Optional[Path] = None,
    default_color=(0, 255, 0),  # green is the default
    dont_show: bool = False,
):
    """
    Prints an image with points drawn on it in the iterm2 terminal.
    Optionally saves the image to a file.
    y coordinates grows down.
    """
    name_to_xy = {}
    for index, point in enumerate(xys):
        x, y = point
        name = f"{index}"
        name_to_xy[name] = [x, y]
    
    prii_named_xy_points_on_image(
        name_to_xy=name_to_xy,
        image=image,
        output_image_file_path=output_image_file_path,
        default_color=default_color,
        dont_show=dont_show
    )
    

    
    
