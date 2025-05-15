import argparse
import sys
from pathlib import Path

from colorama import Fore, Style
from extract_alpha_channel import extract_alpha_channel

usage_message =   """
Usage:
    extract_alpha <source_of_alpha.png>  <where_to_output_alpha.png>
"""

def extract_alpha_cli_tool():
    """
    implements extract_alpha.
    """
    argp = argparse.ArgumentParser()
    argp.add_argument("paths", type=str, nargs="+", help="path to the image with the alpha channel")
    argp.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")

    opt = argp.parse_args()
    paths = opt.paths
    assert len(paths) in [1, 2], usage_message
    dont_ask_for_confirmation_just_kill = opt.yes

    alpha_path = Path(paths[0]).resolve()
    assert alpha_path.is_file(), f"No such file {alpha_path}"

    
    if len(paths) == 2:
        save_path = Path(paths[1]).resolve()
    else:
        save_path = alpha_path  # dangerous, but it's what the user asked for
        # confirm if not forced:
        if not dont_ask_for_confirmation_just_kill:
            answer = input(f"{Fore.YELLOW}going to overwrite {save_path}, are you sure? (y/N){Style.RESET_ALL}")    
            if answer.lower() != "y":
                print("aborting")
                sys.exit(1)
   
    extract_alpha_channel(
        alpha_path=alpha_path,
        save_path=save_path,
    )


