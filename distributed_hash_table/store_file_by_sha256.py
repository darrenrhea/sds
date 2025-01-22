from s3_file_uri_exists import (
     s3_file_uri_exists
)
import sys
from pathlib import Path
import shutil
import subprocess
import textwrap
from hash_tools import sha256_of_file
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from what_computer_is_this import (
     what_computer_is_this
)


def store_file_by_sha256(
    file_path: Path
):
    """
    Saves the file in many places such as locally
    in the sha256 directory, and on the server lam,
    in s3, in cloudflare R2, backblaze, etc.
    It is named its own sha256 hash,
    and you will need that sha256 hash digest as the
    "key" to recover it, so this is a
    "content-addressable" distributed hash table.
    """
    file_path = file_path.resolve()
    computer_name = what_computer_is_this()
    shared_dir = get_the_large_capacity_shared_directory()
    sha256_local_cache_dir = shared_dir / "sha256"

    if not file_path.exists():
        print(f"{file_path} does not exist!")
        sys.exit(1)

    if file_path.is_dir():
        print("That is a directory!")
        sys.exit(1)

    if not file_path.is_file():
        print("That is not a regular file!")
        sys.exit(1)

    the_hash = sha256_of_file(file_path)
    print("The sha256 of the file:")
    print(f"    {file_path}")
    print("is:")
    print(f"    {the_hash}")

    the_extension = file_path.suffix  # what the extension should be 
    print(f"The extension is '{the_extension}'")

    newbasename = f"{the_hash}{the_extension}"

    d01 = the_hash[0:2]
    d23 = the_hash[2:4]
    d45 = the_hash[4:6]
    d67 = the_hash[6:8]

    new_dir = sha256_local_cache_dir / d01 / d23 / d45 / d67
    new_dir.mkdir(parents=True, exist_ok=True)
    storage_location = new_dir / newbasename
    
    if storage_location.is_file():
        print(f"{storage_location} already exists!")
        the_previous_hash = sha256_of_file(storage_location)
        print(
            textwrap.dedent(
                f"""\
                The sha256 of the file:
                    {storage_location}
                is:
                    {the_hash}
                """
            )
        )
        if the_previous_hash != the_hash:
            print(f"ERROR: The file {storage_location} already exists and has a different sha256 hash than it claims to!")
            sys.exit(1)
        else:
            print("We already have a local file which we just confirmed has this sha256 hash, so no need to store it locally again.")
    else:
        print(f"Copying {file_path} to {storage_location}")
        shutil.copy(src=file_path, dst=storage_location)
        print(f"now stored locally at {storage_location}")

        # local permissions get transported to the server, which causes problems when they aren't 0664
        subprocess.run(
            args=[
                "chmod",
                "0664",
                str(storage_location)
            ]
        )

    ssh_to_places = False
    if ssh_to_places:
        if computer_name != "lam":
            subprocess.run(
                args=[
                    "rsync",
                    str(storage_location),
                    "lam:/mnt/nas/volume1/videos/sha256/",
                ]
            )

        subprocess.run(
            args=[
                "rsync",
                str(storage_location),
                "zeus:/mnt/data/www/sha256/",
            ]
        )
    # TODO: check if there is already a s3 uri with the name.  If so, do nothing.
    
    s3_file_uri = f"s3://awecomai-shared/sha256/{newbasename}"

    # TODO: get the head object of the s3 file and check if the sha256 is the same
    if s3_file_uri_exists(s3_file_uri=s3_file_uri):
        print("It was already stored in s3.")
    else:
        print("We need to upload it to s3 because it is not there yet.")

        # TODO: upload it in such a manner that s3 puts the sha256 on the metadata of s3:
        subprocess.run(
            args=[
                "aws",
                "s3",
                "cp",
                str(storage_location),
                "s3://awecomai-shared/sha256/"
            ]
        )

    print(
        textwrap.dedent(
            f"""\
            Anyone could get a copy from s3 via:
            aws s3 cp s3://awecomai-shared/sha256/{newbasename} .
            """
        )
    )


    # print("available by clicking on:")
    # print(f"https://draa.cc/sha256/{newbasename}")
    
    return the_hash


