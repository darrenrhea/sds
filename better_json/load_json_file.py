from io import TextIOWrapper
import textwrap
from typing import Union
from pathlib import Path

from colorama import Fore, Style
import pyjson5


def load_json_file(
    path_or_string_or_fp: Union[str, Path, TextIOWrapper]
) -> Union[dict, list]:
    """
    Loads a JSON or JSON5 or JSON with Comments file into RAM/memory all at once.
    Works with JSON5, JSON with comments, and, of course, plain old JSON.
    TODO: draa add JSON Lines (.jsonl) support.
    """
    assert (
        isinstance(path_or_string_or_fp, str)
        or
        isinstance(path_or_string_or_fp, Path)
        or
        isinstance(path_or_string_or_fp, TextIOWrapper)
    ), f"load_json_file takes in only a Path or a string that can be interpreted as a Path or a TextIO file-object, but you gave it {path_or_string_or_fp}"


    if isinstance(path_or_string_or_fp, TextIOWrapper):
        fp = path_or_string_or_fp
    else:
        p = Path(path_or_string_or_fp).expanduser().resolve()
        fp = open(p, "r")
    # fp is now a file handle.
    try:    
        jsonable = pyjson5.decode_io(fp, None, False)
    except Exception as e:
        print(
            textwrap.dedent(
                f"""
                {Fore.RED}
                Error: {e}
                {Style.RESET_ALL}
                """
            )
        )
        if not isinstance(path_or_string_or_fp, TextIOWrapper):
            print(
                textwrap.dedent(
                    f"""\
                    check if the file
                    
                        {p}
                    
                    is valid JSON5.
                    """
                )
            )
        raise e
    
    return jsonable
