import sys
from get_the_large_capacity_shared_directory import (
    get_the_large_capacity_shared_directory
)

def shared_dir_cli_tool():
    if len(sys.argv) > 1:
        print(f"{get_the_large_capacity_shared_directory(computer_name=sys.argv[1])}", end="")
        exit(1)
    else:
        print(f"{get_the_large_capacity_shared_directory()}", end="")
