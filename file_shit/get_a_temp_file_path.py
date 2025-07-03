from pathlib import Path
import tempfile


def get_a_temp_file_path(
    suffix: str
) -> Path:
    """
    Sometimes you want a temporary file with a specific suffix.
    suffix should be "" if you don't care, or ".jpg", ".png", ".mp4", etc.
    You are responsible for deleting it,
    although it is probably culled whenever
    the system reboots.
    """
    fp = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    )
    return Path(fp.name).resolve()




