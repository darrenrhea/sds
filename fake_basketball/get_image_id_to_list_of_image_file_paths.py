from functools import lru_cache


@lru_cache(maxsize=None)
def get_image_id_to_list_of_image_file_paths(
    folder
) -> dict:
    """
    Say you have a folder with image files in it at top level:
    ~/myfolder/
        a.jpg
        b.png
    
    This would return a dictionary like this:
        {
            "a": [
                Path("~/myfolder/a.jpg")
            ],
            "b": [
                Path("~/myfolder/b.png")
            ]
        }
    """
    
    assert folder.is_dir(), f"ERROR: {folder=} is not a directory."
    
    ad_name_to_list_of_file_paths = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix in [".jpg", ".png"]:
            ad_name_to_list_of_file_paths[p.stem] = [p, ]
         
    return ad_name_to_list_of_file_paths

