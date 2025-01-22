
from typing import Optional


def str_to_maybe_int(s: str) -> Optional[int]:
    """
    Converts a string to an int, if possible,
    otherwise return None.
    """
    assert (
        isinstance(s, str)
    ), f"You promised s would be a str, but it was {type(s)=}, {s=}"
    try:
        return int(s)
    except ValueError:
        return None