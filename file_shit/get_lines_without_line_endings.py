from pathlib import Path
from colorama import Fore, Style, init
import io
from typing import List


def get_lines_without_line_endings(
    file_path: Path
) -> List[str]:
    """
    To the extent that someone uses \r or \r\n,
    we want to replace that with the unix/linux standard \n,
    so this forgetting the line endings is a good thing.
    This is a common operation, so I made a function for it.
    https://stackoverflow.com/questions/39921087/a-openfile-r-a-readline-output-without-n
    """
    assert (
        file_path.resolve().is_file()
    ), f"{Fore.RED}ERROR: {file_path} is not an extant file.{Style.RESET_ALL}"

    lines_without_line_endings = []
    with io.open(file_path, "r", newline=None) as fd:
        for line in fd:
            without_line_ending = line.replace("\n", "")
            lines_without_line_endings.append(without_line_ending)

    return lines_without_line_endings
