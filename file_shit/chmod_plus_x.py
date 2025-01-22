import os
from pathlib import Path
import stat

def chmod_plus_x(path: Path) -> None:
    """
    Make a file path executable
    """
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR)


