from remote_file_exists import (
     remote_file_exists
)
import subprocess
from pathlib import Path


def download_file_via_rsync(
    src_machine: str,
    src_file_path: Path,
    dst_file_path: Path,
    verbose: bool = False
) -> bool:
    """
    Given the string name of an sshable-machine-user-account-pair
    that is defined in your ~/.ssh/config,
    and given the absolute path to a file on that remote machine,
    and given a local absolute path to save the file to,
    this will download the file to the local machine via rsync.
    """
    assert isinstance(src_machine, str), f"{src_machine=} is not a string: it has type {type(src_machine)}"

    assert src_file_path.is_absolute()
    assert dst_file_path.is_absolute(), f"{dst_file_path=} is not absolute.  We need an absolute path since resolving relative paths would require knowing the current working directory."
    
    assert (
        src_file_path.suffix != ""
    ), f"{src_file_path=} has no file extension, like .jpg or .png.  We need a suffix to feel confident you have not given us a directory."
    
    assert (
        dst_file_path.suffix != ""
    ), f"{dst_file_path=} has no file extension, like .jpg or .png.  We need a suffix to feel confident you have not given us a directory."

    
    connection_succeeded, is_file = remote_file_exists(
        sshable_abbrev=src_machine,
        remote_abs_file_path_str=str(src_file_path),
    )
    
    assert (
        connection_succeeded
    ), f"Could not connect to {src_machine=}"

    assert (
        is_file
    ), f"{src_file_path=} is not an extant file on {src_machine=}.  We need a file to download."

    args = [
        "rsync",
        "-P",
        f"{src_machine}:{str(src_file_path)}",
        str(dst_file_path)
    ]
    
    for arg in args:
        assert isinstance(arg, str), f"{arg} of type {type(arg)} is not a string"
    
    if verbose:
        print(" ".join(args))

    subprocess.run(
        args=args,
        cwd=Path.cwd(),
        capture_output=True,
    )

