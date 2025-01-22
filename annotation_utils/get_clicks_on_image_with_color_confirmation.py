import textwrap
from prii import (
     prii
)

from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)

from colorama import Fore, Style

from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)

from get_clicks_on_image import ( 
     get_clicks_on_image
)

from pathlib import Path
import numpy as np

 
def get_clicks_on_image_with_color_confirmation(
    image_path: Path,
    instructions_str: str,
):
    """
    This function will prompt the user to click on the image until they
    press spacebar. It will then show the points on the image
    and ask if the user is happy with the result,
    at which point they can agree or say no and redo.
    Return a list of xy points in pixel coordinates where y grows down,
    i.e. the typical image coordinate system for rasterized images in CS.
    """
    while True:
        print(
            textwrap.dedent(
                f"""\
                {Fore.YELLOW}
                {instructions_str}
                Then press spacebar to finish.
                {Style.RESET_ALL}
                """
            )
        )
        points = get_clicks_on_image(
            image_path=image_path,
            rgba_hwc_np_u8=None,
            instructions_string="press spacebar to finish."
        )

        if len(points) == 0:
            print("You were supposed to click points, but you clicked 0 times!. Try again.")
            continue
        
        name_to_xy = {
            f"{i}": point
            for i, point in enumerate(points)
        }
        
        prii_named_xy_points_on_image(
            image=image_path,
            name_to_xy=name_to_xy
        )

        rgb_np_u8 = open_as_rgb_hwc_np_u8(
            image_path=image_path
        )

        num_clicks = len(points)
        r = int(np.ceil(np.sqrt(num_clicks)))

        rgb_values = np.zeros(
            shape=(r, r, 3),    
            dtype=np.uint8
        )

        for i in range(r):
            for j in range(r):
                if i * r + j < num_clicks:
                    x, y = points[i * r + j]
                    x = int(x)
                    y = int(y)
                    print(f"{x=}")
                    rgb_values[i, j, :] = rgb_np_u8[y, x, :]
        
        bigger = np.round(rgb_values).repeat(200, axis=0).repeat(200, axis=1)
        prii(bigger, caption="Here are the colors you clicked on:")


        ans = input("type y if you happy with this, or s to skip this one, or n to redo: (N/s/y)")

        if ans.lower() == "y":
            break

        if ans.lower() == "s":
            return None
        
        print("Try again please:")
    
    return points
