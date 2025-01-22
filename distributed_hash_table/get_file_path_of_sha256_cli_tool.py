from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import argparse


def get_file_path_of_sha256_cli_tool():
    argp = argparse.ArgumentParser()
    argp.add_argument("sha256", type=str)
    args = argp.parse_args()
    sha256 = args.sha256
    sha256 = sha256.lower().replace("-", "")

    local_path = get_file_path_of_sha256(
        sha256=sha256,
        check=True
    )
    
    print(str(local_path))


