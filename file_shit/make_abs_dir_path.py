
from pathlib import Path
from typing import Union


def make_abs_dir_path(
    x: Union[str, Path]
) -> Path:
    """Converts a string or Path object to an absolute path."""
    if isinstance(x, str):
        assert str[-1] == "/", "make_abs_dir_path requires that any input string end with a forward slash to indicate a directory."
        return Path(x).resolve().expanduser()
    elif isinstance(x, Path):
        return x.resolve().expanduser()
    else:
        raise TypeError(
            f"make_abs_dir_path requires a str or Path, but you gave {x} of {type(x)}"
        )
