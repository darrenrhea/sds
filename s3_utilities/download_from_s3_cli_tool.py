from download_this_s3_file_uri_to_this_file_path import (
     download_this_s3_file_uri_to_this_file_path
)
import sys
from colorama import Fore, Style
from hash_tools import sha256_of_file
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


def download_from_s3_cli_tool():
    if len(sys.argv[1]) < 2:
        print(f"{Fore.RED}You must provide a sha256 to download from s3.{Style.RESET_ALL}")
        sys.exit(1)
    
    sha256 = sys.argv[1]
    assert len(sha256) == 64, f"{sha256} is not a sha256, it should be lowercase hexidecimal and 64 characters long."

    for c in sha256:
        assert c in "0123456789abcdef", f"{sha256} is not a sha256, it should be lowercase hexidecimal and 64 characters long."
    
    s3_file_uri = f"s3://awecomai-shared/sha256/{sha256}"

    shared_dir = get_the_large_capacity_shared_directory()
    sha256_dir = shared_dir / "sha256"

    file_path = sha256_dir / sha256

    if file_path.is_file():
        print(f"{Fore.RED}{file_path} already exists, not downloading.")
        sys.exit(1)

    print(f"downloading to: {file_path}")

    download_this_s3_file_uri_to_this_file_path(
        s3_file_uri=s3_file_uri,
        file_path=file_path
    )

    hexidecimal_sha256 = sha256_of_file(file_path=file_path)
    assert (
        hexidecimal_sha256 == sha256
    ), f"download is corrupt {hexidecimal_sha256} != {sha256}"
   
