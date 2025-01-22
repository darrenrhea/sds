from pathlib import Path
import subprocess
from colorama import Fore, Style
from typing import Optional
import sys

def uname_without_flags() -> str:
    """
    A function that returns the output of
    running the 'uname' command without any flags.
    Usually the output is in the doubleton set {"Darwin", "Linux"},
    so often that we assert this.
    """
    # Run subprocess to execute the 'uname -n' command:
    operating_system = subprocess.run(
        args=[
            'uname',
        ],
        capture_output=True,
        text=True
    ).stdout.strip()
    assert (
        operating_system in {"Darwin", "Linux"}
    ), f"Operating system {operating_system} is not in the set of valid operating systems: {'Darwin', 'Linux'}"
    return operating_system

