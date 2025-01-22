from pathlib import Path
from typing import Union


def unexpanduser(
    p: Union[Path, str]
) -> str:
    """
    This is kinda the inverse function of os.path.expanduser or Path.expanduser.

    We strongly prefer to use absolute paths internally,
    but we want to display paths to the user in a way that is as similar as possible
    to what they might personally type into the shell,
    i.e. ~/foo.txt instead of /home/username/foo.txt.
    """
    p = Path(p)
    if p == Path.home():
        return "~"
    if not p.is_absolute():
        return str(p)
    home = Path("~").expanduser()
    if p.is_relative_to(home):
        rel_path = p.relative_to(home)
        return f"~/{rel_path!s}"
    else:
        return str(p)
  