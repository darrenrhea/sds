import time
from colorama import Fore, Style
from list_all_s3_keys_in_this_bucket_with_this_prefix import (
     list_all_s3_keys_in_this_bucket_with_this_prefix
)
import sys
import textwrap
from pathlib import Path
from typing import Optional
from hash_tools import sha256_of_file
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from download_sha256_from_s3 import (
     download_sha256_from_s3
)


def get_file_path_of_sha256(
    sha256: str,
    check: bool = False  # by default we don't check the sha256 hash
) -> Optional[Path]:
    """
    TODO: should we allow it to store, for example, json5.gz files,
    where the sha256 is calculated on the uncompressed file?
    Or even the canonicalized json file?
    Ditto lossless image compression like PNGs.

    As long as the file is either cached locally or available in s3,
    and you know a prefix of the file's sha256
    which is long enough to specify only one file,
    you can get a local path to it.

    This will try downloading it to the local machine if it doesn't exist.

    This will first look for the file in the local computer's sha256 directory.
    If it finds it, wonderful,
    returns an absolute file Path to it on the local machine, possibly checking the sha256 hash.

    Failing that, it will then it will try downloading it from s3
    to the local machine if it doesn't exist locally.

    If it can't even find it on s3, it will return None as a sign of failure.
    """
    assert isinstance(sha256, str), f"ERROR: sha256 must be a string!, but you gave {sha256=} which has {type(sha256)=}"
    
    assert (
        len(sha256) >= 8
    ), "We need, at a minimum at least 8 hexidecimal characters of the sha256 hash!"
    for c in sha256:
        assert c in "0123456789abcdef", "ERROR: that is not a valid sha256 hash!"
    
    the_sha256_hash = sha256
    shared_dir = get_the_large_capacity_shared_directory()
    sha256_local_cache_dir = shared_dir / "sha256"
    
    d01 = sha256[0:2]
    d23 = sha256[2:4]
    d45 = sha256[4:6]
    d67 = sha256[6:8]

    sub_dir = sha256_local_cache_dir / d01 / d23 / d45 / d67

    exists_locally = False
    while True: 
        if not sub_dir.is_dir():
            print(f"Note: The subdirectory:\n{sub_dir}\ndoes not even exist, so we definitely do not have the file locally!")
            break

        candidates = []
        for p in sub_dir.iterdir():
            if not p.is_file():
                continue
            if p.stem.startswith(the_sha256_hash):
                candidates.append(p)
                local_path = p
        
        if len(candidates) == 0:
            break
        if len(candidates) > 1:
            print("ERROR: There are more than one file with the same sha256 hash:")
            for c in candidates:
                print(c)
            sys.exit(1)

        local_path = candidates[0]
        assert local_path.is_absolute()

        if local_path.exists():
            exists_locally = True
            if check:
                # print(f"Checking the sha256 hash of the file:\n{local_path}")
                recalculated_sha256 = sha256_of_file(local_path)
                if recalculated_sha256[:len(the_sha256_hash)] != the_sha256_hash:
                    for index, c in enumerate(the_sha256_hash):
                        assert recalculated_sha256[index] == the_sha256_hash[index]
                    print(
                        f"ERROR: The file:\n{local_path}\n"
                        f"exists locally, but it does not have correct sha256 that it claims:\n[{the_sha256_hash}]\n"
                        f"it actually comes out to be:\n[{recalculated_sha256}]!")
                    sys.exit(1)
            break

    if not exists_locally:
        start = time.time()
        keys = list_all_s3_keys_in_this_bucket_with_this_prefix(
            bucket="awecomai-shared",
            prefix=f"sha256/{the_sha256_hash}"
        )
        end = time.time()
        print(f"list_all_s3_keys_in_this_bucket_with_this_prefix took {end - start} seconds")
        print(f"{keys=}")
        if len(keys) == 0:
            print(
                textwrap.dedent(
                    f"""\
                    {Fore.YELLOW}
                    Note: The file with sha256 hash
                    
                    {the_sha256_hash}
                    
                    does not exist in s3.
                    
                    You can try for instance:
    
                    aws s3 ls s3://awecomai-shared/sha256/{the_sha256_hash}

                    and you will see that it does not exist.
                    {Style.RESET_ALL}
                    """
                )
            )
            return None
        elif len(keys) > 1:
            print("ERROR: There are more than one file is with the same sha256 hash:")
            for k in keys:
                print(k)
            sys.exit(1)
     
        key = keys[0]
        file_name = key.split("/")[-1]
        full_length_sha256 = file_name.split(".")[0]
        extension = file_name[len(full_length_sha256):]
        
        assert len(full_length_sha256) == 64
        assert full_length_sha256[:len(the_sha256_hash)] == the_sha256_hash

        local_dir = sha256_local_cache_dir / d01 / d23 / d45 / d67
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"{full_length_sha256}{extension}"
        # local_path does not exist, download it from s3:    
        start = time.time()    
        success = download_sha256_from_s3(
            sha256=full_length_sha256,
            extension=extension,
            destination_file_path=local_path,
            verbose=True
        )
        end = time.time()
        print(f"download_sha256_from_s3 took {end - start} seconds")
        if not success:
            print(f"Note: The file:\n{local_path}\n"
                  "does not exist locally, nor could we download it from s3!")
            return None
    
    return local_path



