from make_from_to_mapping_array import (
     make_from_to_mapping_array
)
from pathlib import Path




ad_to_color_map = {
    "denizbank": {
        "led_png_path": Path("~/r/ads_winnowed/denizbank/00349.png").expanduser(),
        "color_map": {
            "denizblue": {
                "is": [18, 208, 216], "should_be": [86, 189, 235]
            },
            "denizwhite": {
                "is": [253, 253, 253], "should_be": [226, 252, 255]
            },
            "denizred": {
                "is": [179, 1, 59], "should_be": [179, 84, 120]
            }
        }
    },

    "skweek_00181": {
        "led_png_path": Path("~/r/munich_led_videos/SKWEEK.COM/skweek_f28057e23/1016x144/00181.png").expanduser(),
        "color_map": {
            "orange": {
                "where": "above the E of Happens",
                "is": (253, 136, 31),
                "ishexis": "#fd881f",
                "should_be": (255, 105, 68)
            },
            "purple": {
                "where": "the purple surrounding the word BASKETBALL",
                "is": (113, 13, 229),
                "should_be": (111, 0, 254)
            },
            "white": {
                "where": "white font top of letter A",
                "is": (255, 252, 255),
                "should_be": (226, 254, 254),
            },
             "brown": {
                "where": "brown groove of basketball under the second P",
                "is": (67, 26, 0),
                "should_be": (95, 70, 76),
            },
            "orangestripes": {
                "where": "middle of stripe over S of happens",
                "is": (255, 96, 14),
                "should_be": (245, 85, 75),
            },
            "orangestripe2": {
                "where": "middle of stripe over last L of basketball",
                "is": (252, 73, 8),  # checked
                "should_be": (239, 69, 82),  # checked
            },
            
        }
    }
}


def get_rgb_from_to_mapping_array():
    """
    TODO: Make this work by clicks on the images.
    """
    color_map = ad_to_color_map["skweek_00181"]["color_map"]
    color_names = sorted(list(color_map.keys()))
    rgb_from_to_mapping_array = make_from_to_mapping_array(
        color_names=color_names,
        color_map=color_map
    )
    print("rgb_from_to_mapping_array=")
    print(rgb_from_to_mapping_array)
    return rgb_from_to_mapping_array
