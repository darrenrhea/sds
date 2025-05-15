from pathlib import Path
import argparse


from pric import (
     pric
)


def pric_cli_tool():
    argp = argparse.ArgumentParser()
    # positional arguments are automatically required:
    argp.add_argument("original_path", type=str)
    argp.add_argument("mask_path", type=str)
    argp.add_argument("--invert", action="store_true")
    argp.add_argument("--saveas", type=str, default=None)
    opt = argp.parse_args()
    original_path = Path(opt.original_path).resolve()
    mask_path = Path(opt.mask_path).resolve()
    assert original_path.is_file(), f"No such file {original_path}"
    assert mask_path.is_file(), f"No such file {mask_path}"
    if opt.saveas is not None:
        saveas = Path(opt.saveas).resolve()
    else:
        saveas = None
    pric(
        rgb_path=original_path,
        alpha_path=mask_path,
        invert=opt.invert,
        saveas=saveas
    )
    

if __name__ == "__main__":
    pric_cli_tool()
