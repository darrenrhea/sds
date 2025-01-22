import os
import hashlib


def sha512_256_of_file(file_path) -> str:
    """
    Calculates the sha512_256 of the file saved at file_path.
    """
    m = hashlib.new("sha512_256")
    with open(os.path.expanduser(file_path), "rb") as fptr:
        for chunk in iter(lambda: fptr.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()

