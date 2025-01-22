from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from print_image_in_iterm2 import print_image_in_iterm2
import sys
from pathlib import Path

def show_help():
    print("This is a program to print image files within the iterm2 terminal.\n")
    print("Usage:\n\n    pri filename\n")
    print("or:\n\n    cat filename | pri -")
    exit(1)


def _read_binary_stdin():
    source = sys.stdin.buffer
    return source.read()


def pri_cli_tool():
    filename = None
    data = None

    if len(sys.argv) != 2:
        show_help()

    if sys.argv[1] != '-':
        filename_or_sha256 = sys.argv[1]

        is_prefix_of_sha256 = False
       
        is_prefix_of_sha256 = all(
            c in "0123456789abcdef-" for c in filename_or_sha256
        )
        if is_prefix_of_sha256:
            sha256 = filename_or_sha256.replace("-", "")
            file_path = get_file_path_of_sha256(
                sha256=sha256,
                check=True
            )
            if file_path is None:
                print(f"SHA256 {filename_or_sha256} not found")
                exit(1)
        else:
            file_path = Path(filename_or_sha256).resolve()
            if not file_path.is_file():
                print(f"{file_path} does not exist")
                exit(1)
        assert (
            file_path.is_file()
        ), f"{file_path} does not exist"
        print_image_in_iterm2(image_path=file_path)
    else:
        data = _read_binary_stdin()
        print_image_in_iterm2(data=data)
    if not file_path and not data:
        show_help()

