from pathlib import Path
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from is_sha256 import (
     is_sha256
)


def gfpfwmbasoafoafps_get_file_path_from_what_might_be_a_sha256_of_a_file_or_a_file_path_str(
    s: str
) -> Path:
    """
    People may want to give a file path directly at a string
    or they may want to give the sha256 of the file.
    """
    if is_sha256(s):
        the_sha256_of_the_file = s
        file_path = get_file_path_of_sha256(
            sha256=the_sha256_of_the_file
        )
        return file_path
    else:
        file_path = Path(s).resolve()
        if not file_path.is_file():
            raise Exception(f"Error:\n    {file_path}\nis not an extant file!")
        return file_path
    
