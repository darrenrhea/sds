from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from typing import Union
from pathlib import Path
import better_json as bj


def ltjfthts_load_the_jsonlike_file_that_has_this_sha256(
    sha256_of_the_jsonlike_file: str,
    check: bool = True
) -> Union[dict, list]:
    """
    Quite often, we know the sha256 hash of a JSON or JSON5 or JSONLines file,
    and want to download it locally then open it.
    """
    file_path = (
        get_file_path_of_sha256(
            sha256=sha256_of_the_jsonlike_file,
            check=check
        )
    )
    assert isinstance(file_path, Path)
    assert file_path.is_file()
    assert file_path.suffix in {".json", ".json5", ".jsonc"}
    jsonable = bj.load(file_path)
    return jsonable