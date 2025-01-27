from colorama import Fore, Style
import sys


def print_red(s: str):
    """
    print in red.
    """
    print(f"{Fore.RED}{s}{Style.RESET_ALL}", file=sys.stderr)
