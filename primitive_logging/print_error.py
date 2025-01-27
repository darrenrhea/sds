import sys
from colorama import Fore, Style

def print_error(s: str):
    print(f"{Fore.RED}WARNING: {s}{Style.RESET_ALL}", file=sys.stderr)
    