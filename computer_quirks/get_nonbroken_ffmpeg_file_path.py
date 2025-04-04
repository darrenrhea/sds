from what_computer_is_this import (
     what_computer_is_this
)
from pathlib import Path


def get_nonbroken_ffmpeg_file_path() -> Path:
    name = what_computer_is_this()
    if name in ["jerry", "appa", "morty", "lam"]:
        nonbroken_ffmpeg_file_path = Path("/usr/bin/ffmpeg")
    elif name in ["korra", "aang", "squanchy"]:
        nonbroken_ffmpeg_file_path = Path("/opt/homebrew/bin/ffmpeg")
    else:
        raise Exception(f"computer_quirks doesn't know where the non-broken ffmpeg executable file path is for the computer named {name}")
    assert(
        nonbroken_ffmpeg_file_path.is_file()
    ), f"ERROR: {nonbroken_ffmpeg_file_path=} is not an extant file"
    return nonbroken_ffmpeg_file_path