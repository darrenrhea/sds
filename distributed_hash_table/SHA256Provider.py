from typing import Optional
from pathlib import Path
from sha256_of_file import (
     sha256_of_file
)

class SHA256Provider(object):
    """
    This is intended to memoize or "index"
    the sha256 of a bunch of files,
    Internally, the key is the resolved absolute path of the file converted to string,
    such as what realpath gives you.
    """
    def __init__(
        self,
        map_from_resolved_abs_file_path_to_sha256
    ):
        self.map_from_resolved_abs_file_path_to_sha256 = map_from_resolved_abs_file_path_to_sha256
    
    def get_sha256(
        self,
        file_path: Optional[Path]
    ):
        if file_path is None:
            return None
        assert isinstance(file_path, Path)
        return self.map_from_resolved_abs_file_path_to_sha256.get(
            str(
                file_path.resolve()
            )
        )

    def add_file(
        self,
        file_path: Path
    ) -> None:
        resolved = file_path.resolve()
        assert resolved.is_file()
        key = str(resolved)
        if key in self.map_from_resolved_abs_file_path_to_sha256:
            return
        
        sha256 = sha256_of_file(file_path)
        self.map_from_resolved_abs_file_path_to_sha256[key] = sha256

