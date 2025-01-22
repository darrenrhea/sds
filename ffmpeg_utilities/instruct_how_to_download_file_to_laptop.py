from what_computer_is_this import (
     what_computer_is_this
)
from pathlib import Path


def instruct_how_to_download_file_to_laptop(
    file_path: Path
):
    """
    If you are executing on a remote computer,
    you might want to instruct
    how to download a file to your laptop.
    """
    
    file_path = file_path.resolve()

    # instruct how to download the video(s) to your laptop:

    # Form the ssh alias to rsync download from:
    username_computer = what_computer_is_this()

    print(f"""
    rsync -P '{username_computer}:{file_path}' ~/show_n_tell/'{file_path.name}'
    """
    )

    print(f"""
    open ~/show_n_tell/{file_path.name}
    """
    )

