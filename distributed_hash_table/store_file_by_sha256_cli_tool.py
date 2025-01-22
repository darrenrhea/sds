from store_file_by_sha256 import store_file_by_sha256
import argparse
from pathlib import Path


def store_file_by_sha256_cli_tool():
    argp = argparse.ArgumentParser()
    argp.add_argument("file_path", type=Path)
    args = argp.parse_args()
    store_file_by_sha256(args.file_path)

