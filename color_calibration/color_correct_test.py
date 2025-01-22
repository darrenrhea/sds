from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from make_from_to_mapping_array import (
     make_from_to_mapping_array
)
import numpy as np
from color_correct_in_lab_space import (
     color_correct_in_lab_space
)
from prii import (
     prii
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from pathlib import Path

clip_id = "bay-zal-2024-03-15-mxf-yadif"
video_frame_index_ad_video_frame_index_pairs = [
    (94960, 181),
    (94891, 146),
]

frame_index, ad_frame_index = video_frame_index_ad_video_frame_index_pairs[1]

shared_dir = get_the_large_capacity_shared_directory()
original_name = f"{clip_id}_{frame_index:06d}_original.jpg"
local_frames_dir_path = shared_dir / "clips" / clip_id / "frames"
original_file_path = local_frames_dir_path / original_name
    

uncorrected_file_path = Path(
    ""
).expanduser()

ads = {
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

channel_names = ["red", "green", "blue"]

#choose an ad:
ad_name = "skweek"

data = ads["skweek_00181"]
color_map = data["color_map"]
# the list color_names decides imposes an ordering from_to_map:
color_names = sorted(list(color_map.keys()))
rgb_from_to_mapping_array = make_from_to_mapping_array(
    color_names=color_names,
    color_map=color_map
)
print("rgb_from_to_mapping_array=")
print(rgb_from_to_mapping_array)


uncorrected_samples = []
for color_name in color_names:

    rgb_triplet = list(color_map[color_name]["is"])
    swatch = np.zeros((200, 200, 3), dtype=np.uint8)
    swatch[:, :, :] = rgb_triplet
    uncorrected_samples.append(swatch)

should_be_samples = []
for color_name in color_names:

    rgb_triplet = list(color_map[color_name]["should_be"])
    swatch = np.zeros((200, 200, 3), dtype=np.uint8)
    swatch[:, :, :] = rgb_triplet
    should_be_samples.append(swatch)



uncorrected_file_path = Path(
    f"~/r/ads_winnowed/{ad_name}/{ad_frame_index:05d}.png"
).expanduser()

uncorrected_rgb = open_as_rgb_hwc_np_u8(
    image_path=uncorrected_file_path
)

uncorrected_rgbs = [uncorrected_rgb] + uncorrected_samples

corrected_rgbs = color_correct_in_lab_space(
    rgb_from_to_mapping_array=rgb_from_to_mapping_array,
    uncorrected_rgbs=uncorrected_rgbs
)

corrected_rgb = corrected_rgbs[0]
corrected_samples = corrected_rgbs[1:]

for uncorrected_sample, should_be_sample, corrected_sample in zip(uncorrected_samples, should_be_samples, corrected_samples):
    prii(uncorrected_sample, caption="uncorrected:")
    prii(should_be_sample, caption="should be:")
    prii(corrected_sample, caption="corrected is:")
    print("\n"*5)

prii(uncorrected_rgb, caption="uncorrected:")
prii(corrected_rgb, caption="corrected:")


print(f"{original_file_path!s}")
prii(original_file_path)