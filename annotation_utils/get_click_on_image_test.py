from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from prii_named_xy_points_on_image import (
     prii_named_xy_points_on_image
)
from pathlib import Path

from get_click_on_image import (
     get_click_on_image
)

def test_get_click_on_image_1():
    image_path = Path("~/r/annotation_app/1001x1000.png").expanduser()

    instructions_string = "Click on the image."
    
    click = get_click_on_image(
        image_path=image_path,
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


def test_get_click_on_image_2():
    image_path = Path("~/r/annotation_app/1001x1000.png").expanduser()
    rgb_hwc_np_u8 = open_as_rgb_hwc_np_u8(image_path)

    instructions_string = "Click on the image."
    
    click = get_click_on_image(
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
    # test_get_click_on_image_1()
    test_get_click_on_image_2()