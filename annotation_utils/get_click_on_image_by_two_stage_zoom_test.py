from get_click_on_image_by_two_stage_zoom import (
     get_click_on_image_by_two_stage_zoom
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from pathlib import Path



def test_get_click_on_image_by_two_stage_zoom_1():
    image_path = Path("~/r/nba_ads/sl/2024-SummerLeague_Courtside_2520x126_TM_STILL.png").expanduser()
    rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(image_path)

    instructions_string = "Click on the image."
    
    click = get_click_on_image_by_two_stage_zoom(
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
       
if __name__ == "__main__":
    test_get_click_on_image_by_two_stage_zoom_1()
