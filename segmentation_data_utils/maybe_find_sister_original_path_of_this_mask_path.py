import textwrap
from pathlib import Path
from typing import Optional
import sys

from colorama import Fore, Style


def maybe_find_sister_original_path_of_this_mask_path(
    mask_path: Path
) -> Optional[Path]:
    """
    Although our masks pretty much always end with _nonfloor.png, the original images
    may have different forms like:
    {annotation_id}_original.png
    {annotation_id}_original.jpg
    {annotation_id}.jpg
    """
    assert (
        mask_path.name.endswith("_nonfloor.png")
    ), "ERROR: mask_path must end with _nonfloor.png!"
    annotation_id = mask_path.name[:-len("_nonfloor.png")]

    assert (
        mask_path.is_file()
    ), "ERROR: mask_path must be an extant file! but {mask_path=} does not exist!"
    
    
    # try _original.png first:
    original_path_version1 = mask_path.parent / f"{annotation_id}_original.png"
    original_path = None
    if original_path_version1.is_file():
        original_path = original_path_version1
    else:
        original_path_version2 = mask_path.parent / f"{annotation_id}.jpg"
        if original_path_version2.is_file():
            original_path = original_path_version2
        else:
            original_path_version3 = mask_path.parent / f"{annotation_id}_original.jpg"
            if original_path_version3.is_file():
                original_path = original_path_version3
            else:
                print(
                    textwrap.dedent(
                        f"""\
                        {Fore.RED}
                        Although
                            {mask_path=}
                        exists, none of these 3 possible corresponding original file paths exist:
                            None of

                               {original_path_version1=}
                            
                            nor
                            
                                {original_path_version2=}
                            nor
                            
                                {original_path_version3=}
                            
                        was found!
                        {Style.RESET_ALL}
                        """
                    )
                )
                return None

    if original_path is not None:
        assert original_path.is_file()
    return original_path