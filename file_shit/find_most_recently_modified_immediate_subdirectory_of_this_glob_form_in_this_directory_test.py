from get_a_temp_dir_path import (
     get_a_temp_dir_path
)
from find_most_recently_modified_immediate_subdirectory_of_this_glob_form_in_this_directory import (
     find_most_recently_modified_immediate_subdirectory_of_this_glob_form_in_this_directory
)


def test_find_most_recently_modified_immediate_subdirectory_of_this_glob_form_in_this_directory_1():
    temp_dir = get_a_temp_dir_path()

    a_dir = temp_dir / "aa"
    a_dir.mkdir()

    b_dir = temp_dir / "ab"
    b_dir.mkdir()

    c_dir = temp_dir / "ac"
    c_dir.mkdir()

    new_file  = b_dir / "new_file.txt"
    new_file.write_text("dog")
    assert new_file.is_file()

    most_recent_dir = find_most_recently_modified_immediate_subdirectory_of_this_glob_form_in_this_directory(
        directory=temp_dir,
        glob_form="a*"
    )
    assert most_recent_dir == b_dir, f"{most_recent_dir} != {b_dir}"


if __name__ == "__main__":
    test_find_most_recently_modified_immediate_subdirectory_of_this_glob_form_in_this_directory_1()