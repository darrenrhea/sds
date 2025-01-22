from blake2b_256_of_file import (
     blake2b_256_of_file
)
from sha512_256_of_file import (
     sha512_256_of_file
)
from sha256_of_file import (
     sha256_of_file
)
from git_sha1_of_file import (
     git_sha1_of_file
)
from sha1_of_file import (
     sha1_of_file
)
from md5_of_file import (
     md5_of_file
)
from get_num_bytes_of_file import (
     get_num_bytes_of_file
)
from pathlib import Path


def test_hash_tools():
    with open("dog.txt", "wb") as fp:
        fp.write(b"dog\n")

    file_path = Path("dog.txt")

    # faster = faster_pseudo_sha256_of_file(file_path)

    num_bytes = get_num_bytes_of_file(file_path=file_path)
    assert num_bytes == 4
    print("num_bytes_of_file passed")

    md5sum = md5_of_file(file_path=file_path)
    assert md5sum == "362842c5bb3847ec3fbdecb7a84a8692"
    print("md5_of_file passed")

    sha1 = sha1_of_file(file_path=file_path)
    assert sha1 == "ee8ca7a80229e38588e5a1062a2320c6c372a097"
    print("sha1_of_file passed")

    git_sha1 = git_sha1_of_file(file_path=file_path)
    assert git_sha1 == "18a619c96eeab6fd6995ff280225eb6175171f95"
    print("git_sha1_of_file passed")

    sha256 = sha256_of_file(file_path=file_path)
    assert sha256 == "b6d8423f6d3423aa233428ab590600486926cf3cd673ab5879d0d36e2dab2671"
    print("sha256_of_file passed")

    sha512_256 = sha512_256_of_file(file_path=file_path)
    assert (
        sha512_256 == "6ce3e967d40c8f03a216bd0e6b12a1c05b9ba27b8046d913baacbdc5113afe6c"
    )
    print("sha512_256_of_file passed")

    blake2b_256 = blake2b_256_of_file(file_path=file_path)
    assert (
        blake2b_256 == "f4912bafeadb3a7cdbe3e9f9998dc06074f315521f38edf9479651e1914a7fb8"
    )
    print("blake2b_256_256_of_file passed")


if __name__ == "__main__":
    test_hash_tools()
