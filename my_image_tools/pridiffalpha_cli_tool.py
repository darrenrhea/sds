"""
Sometimes you need to diff the alpha part of two images
that both convey alpha.
"""
from pathlib import Path
import argparse
from see_alpha_channel_differences import see_alpha_channel_differences


def extant_file_path_from_str(s):
    """
    put this in argparse_utils
    """
    extant_file_path = Path(s).resolve()
    if extant_file_path.is_file():
        return extant_file_path
    else:
        raise argparse.ArgumentTypeError(
            f"{extant_file_path} is not a extant file path"
        )
    
def file_path_from_str(s):
    """
    put this in argparse_utils
    """
    extant_file_path = Path(s).resolve()
    return extant_file_path


def pridiffalpha_cli_tool():
    """
    This is the "main" of the command line utility pridiffalpha.
    """
   
    usage = """
Prints the diff between the alpha parts of two images in iterm2.


Usage:
   pridiffalpha old_alpha_source_image (4 or 1 channel png or 1 channel jpg) old_alpha_source_image (4 or 1 channel png or 1 channel jpg)

   Hard to know what you are looking at unless you also provide an image that has the image data to supperimpose the diff on:

   pridiffalpha \
   alphas/hou-lac-2023-11-14_152156_alpha.png 
   ~/hou-lac-2023-11-14_no-atnt_alpha_mattes_temp/hou-lac-2023-11-14_152156_nonfloor.png \
   --rgb \
   ~/hou-lac-2023-11-14_no-atnt_alpha_mattes_temp/hou-lac-2023-11-14_152156.jpg \
   --out temp.png

Pieces that got added to the alpha mask as we changed from old to new shown in green.
Pieces that got removed shown in red.
"""


    argp = argparse.ArgumentParser(
        usage=usage
    )
    argp.add_argument(
        "alpha_source_a_file_path",
        type=extant_file_path_from_str,
        help="the first image a with alpha channel in it"
    )
    argp.add_argument(
        "alpha_source_b_file_path",
        type=extant_file_path_from_str,
        help="the second image b with alpha channel in it"
    )
    argp.add_argument(
        "--rgb",
        default=None,
        type=extant_file_path_from_str,
        help="a source of the color rgb info to superimpose the red/green diff on"
    )
    argp.add_argument(
        "--out",
        default=None,
        type=file_path_from_str,
        help="where to save the diff image"
    )
    argp.add_argument(
        '--print_in_terminal',
        action='store_true',
        help="print the diff image in the terminal esp. iterm2"
    )

    opt = argp.parse_args()
    alpha_source_a_file_path = opt.alpha_source_a_file_path
    alpha_source_b_file_path = opt.alpha_source_b_file_path
    rgb_source_file_path = opt.rgb
    save_file_path = opt.out
    print(f"{save_file_path=}")
    print_in_terminal = opt.print_in_terminal
    print(f"{print_in_terminal=}")

    see_alpha_channel_differences(
        alpha_source_a_file_path=alpha_source_a_file_path,
        alpha_source_b_file_path=alpha_source_b_file_path,
        rgb_source_file_path=rgb_source_file_path,
        save_file_path=save_file_path,
        print_in_terminal=((save_file_path is None) or print_in_terminal)
    )

