from colorama import Fore, Style
import sys


def print_yellow(s: str) -> None:
    """
    Being able to print in color is nice,
    but what is even nicer is to print to stderr rather than stdout.
    """
    print(f"{Fore.YELLOW}{s}{Style.RESET_ALL}", file=sys.stderr)
