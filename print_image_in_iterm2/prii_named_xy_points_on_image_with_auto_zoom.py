from typing import Dict, Optional, Tuple
from convert_to_rgba_hwc_np_u8 import (
     convert_to_rgba_hwc_np_u8
)
import PIL.Image
from prii import (
     prii
)
from pathlib import Path
import numpy as np
from Drawable2DImage import Drawable2DImage


def prii_named_xy_points_on_image_with_auto_zoom(
    name_to_xy: Dict[str, Tuple[int, int]],
    image: str | Path | np.ndarray | PIL.Image.Image,
    output_image_file_path: Optional[Path] = None,
    default_color=(0, 255, 0),  # green is the default
    dont_show: bool = False,
):
    """
    Prints an image with named points on it in the iterm2 terminal.
    Optionally saves the image to a file.
    y grows down
    """
    rgba_hwc_np_u8 = convert_to_rgba_hwc_np_u8(image)
   
    height, width = rgba_hwc_np_u8.shape[:2]
    
    margin = 100
    
    x_max = max([xy[0] for xy in name_to_xy.values()])
    y_max = max([xy[1] for xy in name_to_xy.values()])
    x_min = min([xy[0] for xy in name_to_xy.values()])
    y_min = min([xy[1] for xy in name_to_xy.values()])
    x_max = min(x_max + margin, width)
    y_max = min(y_max + margin, height)
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max = int(x_max)
    y_max = int(y_max)
    x_min = int(x_min)
    y_min = int(y_min)

    # print_yellow(f"{x_min=}, {x_max=}, {y_min=}, {y_max=} debug4748387437839752")
    sub_image = rgba_hwc_np_u8[y_min:y_max, x_min:x_max, :]

    drawable_image = Drawable2DImage(
        rgba_np_uint8=sub_image,
        expand_by_factor=3
    )

    for name, xy in name_to_xy.items():
        x = xy[0] - x_min
        y = xy[1] - y_min

        if len(xy) == 3:
            attributes = xy[2]
        else:
            attributes = {
                "rgb": default_color,
                "size": 10,
            }
        drawable_image.draw_plus_at_2d_point(
            x_pixel=x,
            y_pixel=y,
            rgb=attributes["rgb"],
            size=10,
            text=name
        )
    
    if not dont_show:
        prii(drawable_image.image_pil)
    
    if output_image_file_path is not None:
        assert isinstance(output_image_file_path, Path)
        output_image_file_path.parent.is_dir()
        output_image_file_path = output_image_file_path.resolve()
        drawable_image.save(output_image_file_path=output_image_file_path)
        print(f"saved image to {output_image_file_path}")
    

    
    
