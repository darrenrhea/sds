from what_computer_is_this import (
     what_computer_is_this
)
from pathlib import Path


def get_font_file_path() -> Path:
    name = what_computer_is_this()
    if name in ["aang", "korra"]:
        font_file_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    elif name == "lam":
        font_file_path = Path("/awecom/misc/arial.ttf")
    elif name in ["jerry", "appa", "morty", "rick", "grogu"]:
        font_file_path = Path("/shared/fonts/arial.ttf")
    elif name in ["dockercontainer"]:
        font_file_path = Path(
            "/shared/sha256/82/af/b3/5e/82afb35eda3a52edb10106bcc04af93646384421ded538d38792c1444d816022.ttf"
        )
    else:
        raise Exception(f"computer_quirks doesn't know where the font file path is for the computer named {name}")
    assert font_file_path.is_file(), f"ERROR: {font_file_path=} is not an extant file"
    return font_file_path

