from setuptools import setup

setup(
    name="file_shit",
    version="0.1.0",
    py_modules=[
        "chmod_plus_x",
        "delete_all_files_in_this_folder_that_are_not_in_this_folder",
        "find_most_recently_modified_file_in_this_directory",
        "find_most_recently_modified_immediate_subdirectory_of_this_glob_form_in_this_directory",
        "get_a_temp_dir_path",
        "get_a_temp_file_path",
        "get_lines_without_line_endings",
        "make_abs_dir_path",
        "make_abs_file_path",
        "unexpanduser",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "delete_all_files_in_this_folder_that_are_not_in_this_folder = delete_all_files_in_this_folder_that_are_not_in_this_folder_cli_tool:delete_all_files_in_this_folder_that_are_not_in_this_folder_cli_tool",
        ],
    },
)
