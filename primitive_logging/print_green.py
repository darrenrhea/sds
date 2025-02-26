import sys
from colorama import Fore, Style


def print_green(s: str):
    """
    Being able to print in color is nice,
    but what is even nicer is to print to stderr rather than stdout.
    """
    print(f"{Fore.GREEN}{s}{Style.RESET_ALL}", file=sys.stderr)
