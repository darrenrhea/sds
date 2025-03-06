from print_green import (
     print_green
)
from print_yellow import (
     print_yellow
)
from print_red import (
     print_red
)
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

def store_file_by_sha256_in_s3(
    file_path: Path,
    verbose: bool = False
) -> str:  # debatably there should be a SHA256 type
    """
    Saves the file in s3.
    It is named its own sha256 hash,
    and you will need that sha256 hash digest as the
    "key" to recover it, so this is a
    "content-addressable" distributed hash table.
    """
    file_path = file_path.resolve()

    if not file_path.exists():
        print_red(f"{file_path} does not exist!")
        sys.exit(1)

    if file_path.is_dir():
        print_red("That is a directory!")
        sys.exit(1)

    if not file_path.is_file():
        print_red("That is not a regular file!")
        sys.exit(1)

    the_sha256_hash = sha256_of_file(file_path)
    if verbose:
        print_green("The sha256 of the file:")
        print_green(f"    {file_path}")
        print_green("is:")
        print_green(f"    {the_sha256_hash}")

    the_extension = file_path.suffix  # what the extension should be 
    if verbose:
        print_green(f"The extension is '{the_extension}'")

    newbasename = f"{the_sha256_hash}{the_extension}"
 
    s3_file_uri = f"s3://awecomai-shared/sha256/{newbasename}"

    # TODO: get the head object of the s3 file and check if the sha256 is the same
    if s3_file_uri_exists(s3_file_uri=s3_file_uri):
        print_green("It was already stored in s3.")
    else:
        print_yellow("We need to upload it to s3 because it is not there yet.")

        # TODO: upload it in such a manner that s3 puts the sha256 on the metadata of s3:
        subprocess.run(
            args=[
                "aws",
                "s3",
                "cp",
                str(file_path),
                f"s3://awecomai-shared/sha256/{newbasename}"
            ]
        )

    print_green(
        textwrap.dedent(
            f"""\
            Anyone could get a copy from s3 via:
            aws s3 cp s3://awecomai-shared/sha256/{newbasename} .
            """
        )
    )

    return the_sha256_hash


