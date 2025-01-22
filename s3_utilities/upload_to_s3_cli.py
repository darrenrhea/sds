from upload_file_path_to_s3_file_uri import (
     upload_file_path_to_s3_file_uri
)
import sys
from pathlib import Path
from colorama import Fore, Style
from hash_tools import sha256_of_file


def upload_to_s3_cli():
    """
    """
    if len(sys.argv[1]) < 2:
        print(f"{Fore.RED}You must provide a file path to upload to s3.{Style.RESET_ALL}")
        sys.exit(1)
    
    file_path = Path(sys.argv[1]).resolve()

    assert file_path.is_file(), f"{file_path} is not a file."

    hexidecimal_sha256 = sha256_of_file(file_path=file_path)
    
    s3_file_uri = f"s3://awecomai-shared/sha256/{hexidecimal_sha256}"
    
    print(f"Because the file's SHA256 is {hexidecimal_sha256}, we will upload it to {s3_file_uri}.")

    upload_file_path_to_s3_file_uri(
        file_path=file_path,
        s3_file_uri=f"s3://awecomai-shared/sha256/{hexidecimal_sha256}",
        verbose=True,
        expected_hexidecimal_sha256=hexidecimal_sha256
    )



