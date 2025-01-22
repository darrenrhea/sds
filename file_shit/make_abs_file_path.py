
from pathlib import Path
import textwrap
from typing import Union


def make_abs_file_path(
    x: Union[str, Path]
) -> Path:
    """Converts a string or Path object to an absolute path."""
    # order of expanduser and resolve is important, expanduser must be first.
    if isinstance(x, str):
        abs_file_path = Path(x).expanduser().resolve()
    elif isinstance(x, Path):
        abs_file_path = x.expanduser().resolve()
    else:
        raise TypeError(
            f"make_abs_file_path requires a str or Path, but you gave {x} of {type(x)}"
        )
    
    assert (
        abs_file_path.suffix != ""
    ), textwrap.dedent(
        """\
        You gave make_abs_file_path {x}
        make_abs_file_path requires that the input str or Path
        "seems like a file" in the sense that it ends with a file extension,
        such as .exe, .json, .png, .txt, .csv, etc.
        If you don't like this,
        You can always do it yourself with Path(x).resolve().expanduser()
        """
    )

    assert abs_file_path.parent.is_dir(), textwrap.dedent(
        f"""\
        The parent directory of {abs_file_path} does not currently exist.
        Please make sure that the parent directory exists before using make_abs_file_path.
        """
    )

    return abs_file_path
