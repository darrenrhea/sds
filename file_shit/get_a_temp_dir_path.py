from pathlib import Path
import tempfile

def get_a_temp_dir_path() -> Path:
    """
    You are responsible for deleting it,
    although it is probably culled whenever
    the system reboots.
    """
    return Path(
        tempfile.mkdtemp()
    ).resolve()


