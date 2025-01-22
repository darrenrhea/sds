from prii import (
     prii
)
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=None)
def get_subdir_name_to_list_of_image_file_paths(
    mother_folder
) -> dict:
    """
    This is what decides which rips get stuck into the LED boards.
    Maybe rename get_subdir_name_to_list_of_image_file_paths.
    """
    
    assert mother_folder.is_dir(), f"ERROR: {mother_folder=} is not a directory."
    
    ad_ids = []
    for ad_id in mother_folder.iterdir():
        if ad_id.is_dir():
            ad_ids.append(ad_id.name)
    
  

    # In the NBA, we don't have any ads that don't need color correction because we don't rip ads:
    ad_name_to_list_of_file_paths = {
        ad_id: (
            [
                p for p in (mother_folder / ad_id).glob("*.png")
            ]
            +
            [
                p for p in (mother_folder / ad_id).glob("*.jpg")
            ]
        )
        for ad_id in ad_ids
    }
     
    return ad_name_to_list_of_file_paths

