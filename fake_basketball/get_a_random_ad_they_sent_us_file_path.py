
from pathlib import Path
import numpy as np


def get_a_random_ad_they_sent_us_file_path(
    ad_name_to_paths_that_do_need_color_correction
)-> Path:
    all_ad_names = set()
    
    for key, value in ad_name_to_paths_that_do_need_color_correction.items():
        if len(value) > 0:
            all_ad_names.add(key)
    
    
    ad_name = np.random.choice(list(all_ad_names))

    print(f"From amongst all the ads they sent us (which need color correction) We randomly chose the ad {ad_name}")

    ad_paths_that_do_need_color_correction = ad_name_to_paths_that_do_need_color_correction.get(ad_name, [])

    assert (
        len(ad_paths_that_do_need_color_correction) > 0
    ), f"ERROR: {ad_name=} has no images for it. This should never happen."

    index = np.random.randint(0, len(ad_paths_that_do_need_color_correction))
    ad_file_path = ad_paths_that_do_need_color_correction[index]
    return ad_file_path
