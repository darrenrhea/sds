from get_click_on_subrectangle_of_image import (
     get_click_on_subrectangle_of_image
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from pathlib import Path

from get_click_on_resized_image import (
     get_click_on_resized_image
)


def test_get_click_on_resized_image_1():
    image_path = Path("~/r/nba_ads/sl/2024-SummerLeague_Courtside_2520x126_TM_STILL.png").expanduser()
    rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(image_path)

    instructions_string = "Click on the image."
    
    click = get_click_on_resized_image(
        max_display_width=1800,
        max_display_height=800,
        rgb_hwc_np_u8=rgb_hwc_np_u8,
        instructions_string=instructions_string
    )

    print(click)

    if click is None:
        name_to_xy = {}
        print("The user declined to click.")
    else:
        name_to_xy = {
            "click": click
        }
        print("The user clicked here:")

        prii_named_xy_points_on_image(
            image=image_path,
            name_to_xy=name_to_xy
        )
        i_min = max(0, click[1] - 50)
        i_max = min(rgb_hwc_np_u8.shape[0], click[1] + 50)
        j_min = max(0, click[0] - 50)
        j_max = min(rgb_hwc_np_u8.shape[1], click[0] + 50)
        click = get_click_on_subrectangle_of_image(
            j_min=j_min,
            j_max=j_max,
            i_min=i_min,
            i_max=i_max,
            rgb_hwc_np_u8=rgb_hwc_np_u8,
            instructions_string=instructions_string
        )

        if click is None:
            name_to_xy = {}
            print("The user declined to click.")
        else:
            name_to_xy = {
                "click": click
            }
            print("The user clicked here:")

            prii_named_xy_points_on_image(
                image=image_path,
                name_to_xy=name_to_xy
            )
       

if __name__ == "__main__":
    test_get_click_on_resized_image_1()