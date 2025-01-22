from file_shit import find_most_recently_modified_file_in_this_directory
from file_shit import *
from pathlib import Path
from get_a_temp_dir_path import get_a_temp_dir_path

def test_find_most_recently_modified_file_in_this_directory():
    temp_dir_path = get_a_temp_dir_path()
    a_file_path = temp_dir_path / "a.txt"
    a_file_path.write_text("dog")
    b_file_path = temp_dir_path / "b.txt"
    b_file_path.write_text("cat")

    most_recent_file_path = find_most_recently_modified_file_in_this_directory(
        directory=temp_dir_path,
        predicate=(lambda x: True)
    )

    print(f"{most_recent_file_path=}")
    assert most_recent_file_path == b_file_path, f"{most_recent_file_path} != {b_file_path}" 


if __name__ == "__main__":
    test_find_most_recently_modified_file_in_this_directory()

