from get_clicks_on_image import ( 
     get_clicks_on_image
)

from pathlib import Path


def test_get_clicks_on_image_1():
    image_path = Path(
        "~/r/munich4k_led/DSCF0241_001128.jpg"
    ).expanduser()
    get_clicks_on_image(
        image_path=image_path,
        rgb_hwc_np_u8=None,
        instructions_string="click head then toe. press spacebar to finish."
    )
 
if __name__ == "__main__":
    test_get_clicks_on_image_1()
