import os
import hashlib


def sha1_of_file(file_path):
    "We don't use straight SHA1 because the git variant with the blob length header is more common"
    m = hashlib.sha1()
    with open(os.path.expanduser(file_path), "rb") as fptr:
        for chunk in iter(lambda: fptr.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()

