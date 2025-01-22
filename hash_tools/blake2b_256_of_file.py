
import os
import hashlib


def blake2b_256_of_file(file_path) -> str:
    """
    $ printf "The quick brown fox jumps over the lazy dog" | b2sum -l 256
    01718cec35cd3d796dd00020e0bfecb473ad23457d063b75eff29c0ffa2e58a9  -
    """
    m = hashlib.blake2b(digest_size=32) # 64 bytes = 512 bits, 32 bytes = 256 bits
    with open(os.path.expanduser(file_path), "rb") as fptr:
        for chunk in iter(lambda: fptr.read(4096), b""):
            m.update(chunk)
    return m.hexdigest()
