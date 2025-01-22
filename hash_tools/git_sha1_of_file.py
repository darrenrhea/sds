from get_num_bytes_of_file import (
     get_num_bytes_of_file
)

import os
import hashlib


def git_sha1_of_file(file_path):
    num_bytes = get_num_bytes_of_file(file_path=file_path)
    m = hashlib.sha1()
    chunk = (f"blob {num_bytes}\0").encode("ascii")
    m.update(chunk)
    with open(os.path.expanduser(file_path), "rb") as fptr:
        for chunk in iter(lambda: fptr.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()

