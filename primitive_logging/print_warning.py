import sys
from colorama import Fore, Style

def print_warning(s: str):
    print(
        f"{Fore.YELLOW}WARNING: {s}{Style.RESET_ALL}",
        file=sys.stderr
    )
    