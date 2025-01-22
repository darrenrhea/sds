from setuptools import setup

setup(
    name="computer_quirks",
    version="0.1.0",
    py_modules=[
        "computer_quirks",
        "get_font_file_path",
        "get_nonbroken_ffmpeg_file_path",
        "get_the_large_capacity_shared_directory",
        "uname_dash_n",
        "uname_without_flags",
        "what_computer_is_this",
        "what_os_is_this",
        "whoami",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "better_json",
        "colorama",
        "oyaml",
        "paramiko",
        "pyperclip",
        "requests",
        "via_ssh",
    ],
    entry_points={
        "console_scripts": [
            "shared_dir = shared_dir_cli_tool:shared_dir_cli_tool",
            "what_computer_is_this = what_computer_is_this_cli_tool:what_computer_is_this_cli_tool",
            "what_os_is_this = what_os_is_this_cli_tool:what_os_is_this_cli_tool",
        ],
    },
)
