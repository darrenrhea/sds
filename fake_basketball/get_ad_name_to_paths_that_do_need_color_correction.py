from get_image_id_to_list_of_image_file_paths import (
     get_image_id_to_list_of_image_file_paths
)

from prii import (
     prii
)
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def get_ad_name_to_paths_that_do_need_color_correction(
    verbose: bool = True,
    visualize: bool = True
) -> dict:
    """
    This is what decides which ads they sent over get stuck into the LED boards.
    """

    ads_they_sent_to_us_folder = Path(
        "~/r/nba_ads/summer_league_2024"
    ).expanduser()
    
    ad_name_to_list_of_file_paths = \
    get_image_id_to_list_of_image_file_paths(
        ads_they_sent_to_us_folder
    )

    if verbose:
        print("These are the ads that they sent over to us which we will use to make fake data after color correction, augmentation, etc.")
    for ad_id, paths in ad_name_to_list_of_file_paths.items():
        if verbose:
            print(f"\nFor {ad_id=} they sent over these idealized ads:")
        assert len(paths) > 0, f"ERROR: {paths=} is empty for {ad_id=}."
        for cntr, p in enumerate(paths):
            if verbose:
                print(p)
            assert p.is_file(), f"ERROR: {p=} is not a file."
            if visualize:
                prii(p)
            
  
     
    return ad_name_to_list_of_file_paths

   
if __name__ == "__main__": 
    # this is a test / demo: 
    ad_name_to_paths_that_do_need_color_correction = get_ad_name_to_paths_that_do_need_color_correction(
        verbose=True,
        visualize=True
    )
