
import numpy as np


def get_a_random_adrip_file_path(
    ad_name_to_paths_that_dont_need_color_correction
):
    all_ad_names = set()
    
    for key, value in ad_name_to_paths_that_dont_need_color_correction.items():
        if len(value) > 0:
            all_ad_names.add(key)
    
    
    ad_name = np.random.choice(list(all_ad_names))

    print(f"From amongst all adrips (which don't need color correction) We randomly chose the ad {ad_name}")

    ad_paths_that_dont_need_color_correction = ad_name_to_paths_that_dont_need_color_correction.get(ad_name, [])

    assert (
        len(ad_paths_that_dont_need_color_correction) > 0
    ), f"ERROR: {ad_name=} has no rip PNGs for it."

    index = np.random.randint(0, len(ad_paths_that_dont_need_color_correction))
    ad_texture_png_path = ad_paths_that_dont_need_color_correction[index]
    return ad_texture_png_path
