from uname_without_flags import (
     uname_without_flags
)
from pathlib import Path
from colorama import Fore, Style


def what_os_is_this() -> str:
    """
    A function that returns
    which operating system the executing computer is running
    as a string in ["macos", "linux", "unknown_os"]
    """
    by_home_dir = "unknown_os"
    home_dir = Path("~").expanduser()
    if str(home_dir).startswith("/Users"):
        by_home_dir = "macos"
    elif str(home_dir).startswith("/home"):
        by_home_dir = "linux"
    else:
        raise Exception("I don't know what computer this is.")
    uname = uname_without_flags()
    by_uname = dict(Linux="linux", Darwin="macos").get(uname, "unknown_os")
    if by_home_dir != by_uname:
        print(
             f"{Fore.YELLOW}WARNING OS identification by home dir: {by_home_dir} is not equal to by uname {by_uname}{Style.RESET_ALL}"
        )
    assert by_uname in ["macos", "linux", "unknown_os"]
    return by_uname

