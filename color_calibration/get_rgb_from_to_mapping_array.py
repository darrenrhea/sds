from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from color_print_json import (
     color_print_json
)
from make_from_to_mapping_array import (
     make_from_to_mapping_array
)
from pathlib import Path
import numpy as np
import better_json as bj


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
            "above_SE": {
                "where": "above the S and E of Happens",
                "is": (255, 108, 66),
                "should_be": (255, 120, 55),
            },
            "orange": {
                "where": "above the E of Happens",
                "is": (253, 136, 31),
                "hexidecimal_is": "#fd881f",
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


def get_all_color_correction_data_points():
    """
    This returns all the color correction data points.
    """
    color_correction_data_points_dir = Path(
        "~/r/color_correction_data/color_correction_data_points"
    ).expanduser()
    
    color_correction_data_points = []
    for p in color_correction_data_points_dir.iterdir():
        if not p.is_file():
            continue
        if not p.suffix == ".json5":
            continue
        color_correction_data_point = bj.load(p)
        color_correction_data_points.append(color_correction_data_point)

    return color_correction_data_points


def add_average_domain_and_codomain_colors_to_color_correction_data_point(
    color_correction_data_point
):
    """
    This adds the average domain and codomain colors to the color_correction_data_point.
    """
    domain_points = color_correction_data_point["domain_points"]
    codomain_points = color_correction_data_point["codomain_points"]
    
    domain_image_sha256 = color_correction_data_point["domain_image_sha256"]
    codomain_image_sha256 = color_correction_data_point["codomain_image_sha256"]

    domain_image_file_path = get_file_path_of_sha256(domain_image_sha256)
    
    codomain_image_file_path = get_file_path_of_sha256(codomain_image_sha256)
    
    domain_rgb_np_u8 = open_as_rgb_hwc_np_u8(domain_image_file_path)
    codomain_rgb_np_u8 = open_as_rgb_hwc_np_u8(codomain_image_file_path)

    domain_colors = np.array([
        domain_rgb_np_u8[
            int(np.round(xy[1])),
            int(np.round(xy[0])),
            :
        ]
        for xy in domain_points
    ])

    codomain_colors = np.array([
        codomain_rgb_np_u8[
            int(np.round(xy[1])),
            int(np.round(xy[0])),
            :
        ]
        for xy in codomain_points
    ])

    average_domain_color = [int(round(x)) for x in np.mean(domain_colors, axis=0)]
    average_codomain_color = [int(round(x)) for x in np.mean(codomain_colors, axis=0)]
    description = color_correction_data_point["description"]
    print(f"\n\n{description=}")
    print(f"{average_domain_color=}")
    print(f"{average_codomain_color=}\n")

    color_correction_data_point["average_domain_color"] = average_domain_color
    color_correction_data_point["average_codomain_color"] = average_codomain_color



def get_rgb_from_to_mapping_array(
    color_correction_context_id: str
) -> np.ndarray:
    """
    This builds up a color correction regression context and returns the rgb_from_to_mapping_array.
    """

    color_correction_context = bj.load(
        f"~/r/color_correction_data/color_correction_contexts/{color_correction_context_id}.json5"
    )

    color_print_json(color_correction_context)


    color_correction_data_points = get_all_color_correction_data_points()
    
    # filter down to data points that are relevant to this context:
    color_correction_data_points = [
        x for x in color_correction_data_points
        if x["color_correction_context_id"] == color_correction_context_id
    ]
    for color_correction_data_point in color_correction_data_points:
        add_average_domain_and_codomain_colors_to_color_correction_data_point(color_correction_data_point)

    rgb_from_to_mapping_array = np.array([
        [
            color_correction_data_point["average_domain_color"],
            color_correction_data_point["average_codomain_color"]
        ]
        for color_correction_data_point in color_correction_data_points
    ])

    return rgb_from_to_mapping_array


if __name__ == "__main__":
    color_correction_context_id = "5922a16d-0a20-467e-b72c-6a64ca78fabe"
    get_rgb_from_to_mapping_array(
        color_correction_context_id=color_correction_context_id
    )