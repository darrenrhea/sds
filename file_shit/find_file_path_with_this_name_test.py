from find_file_path_with_this_name import (
     find_file_path_with_this_name
)
from pathlib import Path


def test_find_file_path_with_this_name_1():
    dir_path_to_search = Path("~/r/bos-mia-2024-04-21-mxf_led").expanduser()
    file_name = "bos-mia-2024-04-21-mxf_734500_nonfloor.png"
    ans = find_file_path_with_this_name(
        dir_path_to_search=dir_path_to_search,
        file_name=file_name
    )
    assert ans is not None, f"{ans=} is None."


def test_find_file_path_with_this_name_2():
    dir_path_to_search = Path("~/r/bos-mia-2024-04-21-mxf_led").expanduser()
    file_name = "jibberish37457578387873"
    ans = find_file_path_with_this_name(
        dir_path_to_search=dir_path_to_search,
        file_name=file_name
    )
    assert ans is None, "ERROR: Should have come out None."


if __name__ == "__main__":
    test_find_file_path_with_this_name_1()
    test_find_file_path_with_this_name_2()
    print("find_file_path_with_this_name.py has passed all tests.")