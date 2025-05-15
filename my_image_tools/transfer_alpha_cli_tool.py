import sys
from pathlib import Path
from transfer_alpha_channel import transfer_alpha_channel


def transfer_alpha_cli_tool():
    """
    implements transfer_alpha.
    """
    if len(sys.argv) < 4:
        print(
            """
Usage:
    transfer_alpha <source_of_alpha.png> <source_of_rgb_info.(png|jpg|bmp)> <where_to_output_combined_rgba.png>
"""
        )
        sys.exit(1)

    alpha_path = Path(sys.argv[1]).resolve()
    rgb_path = Path(sys.argv[2]).resolve()
    save_path = Path(sys.argv[3]).resolve()
    assert alpha_path.is_file(), f"No such file {alpha_path}"
    assert rgb_path.is_file(), f"No such file {rgb_path}"

    # if save_path.is_file():
    #     ans = input(
    #         f"{save_path} already exists, are you sure you want to overwrite it? "
    #     )
    #     if ans not in ["yes", "Yes", "y", "Y"]:
    #         print("Stopping")
    #         sys.exit(1)

    transfer_alpha_channel(
        alpha_path=alpha_path, rgb_path=rgb_path, save_path=save_path
    )


