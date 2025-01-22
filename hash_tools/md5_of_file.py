from is_lowercase_md5_checksum import (
     is_lowercase_md5_checksum
)
import os
import hashlib


def md5_of_file(file_path) -> str:
    m = hashlib.md5()
    with open(os.path.expanduser(file_path), "rb") as fptr:
        # the iterator's __next__ method with call fptr.read(4096) again and again
        # until the returned value is b"", at which point it stops.
        for chunk in iter(lambda: fptr.read(4096), b""):
            m.update(chunk)
    md5_str = m.hexdigest()
    assert is_lowercase_md5_checksum(md5_str), "ERROR: md5 checksum is not lowercase hexidecimal or length 32 hexidecimal characters!?"
    return md5_str

