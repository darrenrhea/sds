from get_subdir_name_to_list_of_image_file_paths import (
     get_subdir_name_to_list_of_image_file_paths
)
from prii import (
     prii
)
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def get_ad_name_to_paths_that_dont_need_color_correction() -> dict:
    """
    This is what decides which rips get stuck into the LED boards.
    """

    mother_folder = Path(
        "~/r/nba_ads_that_dont_need_color_correction/summer_league_2024"
    ).expanduser()
    
    ad_name_to_list_of_file_paths = get_subdir_name_to_list_of_image_file_paths(mother_folder)

    print("These are the rips we will use:")
    for ad_id, list_of_file_paths in ad_name_to_list_of_file_paths.items():
        print(ad_id)
        for p in list_of_file_paths:
            print(p)
        print("\n\n\n")
  
     
    return ad_name_to_list_of_file_paths

   
if __name__ == "__main__": 
    # this is a test / demo: 
    ad_name_to_paths_that_dont_need_color_correction = get_ad_name_to_paths_that_dont_need_color_correction()


    print("Rips, i.e. ads that do not need color correction:")

    for ad_name, paths in ad_name_to_paths_that_dont_need_color_correction.items():
        print(f"\n{ad_name} has these rips:")
        assert len(paths) > 0, f"ERROR: {paths=} is empty for {ad_name=}."
        for cntr, p in enumerate(paths):
            print(p)
            assert p.is_file(), f"ERROR: {p=} is not a file."
            prii(p)
            if cntr  == 2:
                break

    
    