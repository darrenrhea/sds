from pathlib import Path
import tempfile

def get_a_temp_file_path(suffix) -> Path:
    """
    You are responsible for deleting it,
    although it is probably culled whenever
    the system reboots.
    """
    fp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    return Path(fp.name).resolve()




