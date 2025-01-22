from remote_directory_exists import (
     remote_directory_exists
)
import subprocess
from pathlib import Path


def upload_file_via_rsync(
    dst_machine: str,
    src_file_path: Path,  # the src_file_path MUST be a file
    dst_file_path: Path,  # the dst_file_path MUST be a file
    verbose: bool = False
) -> bool:
    """
    The source path MUST be a file.
    Because the semantics of rsyncing directories is a bit complicated with respect to whether it:
    1. makes a new subdirectory named the same as the src directory's name
    2. or just moves the contents of the source directory into the destination directory
    we could do much better than this.  
    """
    assert dst_machine in ["lam", "squanchy", "plumbus", "appa"]

    assert src_file_path.is_absolute()
    assert (
        src_file_path.is_file()
    ), f"{src_file_path=} is not an extant file on the executing machine.  We need a file to upload."

    
    assert (
        dst_file_path.suffix != ""
    ), f"{dst_file_path=} has no file extension, like .jpg or .png.  We need a suffix to feel confident you have not given a directory."
    
    assert (
        dst_file_path.is_absolute()
    ), f"{dst_file_path=} is not absolute.  We need an absolute path since resolving relative paths would require knowing the current working directory."

    remote_directory_exists(
        remote_directory_str=str(dst_file_path),
        sshable_abbrev=dst_machine,
        verbose=True
    )


    args = [
        "rsync",
        "-rP",
        str(src_file_path),
        f"{dst_machine}:{str(dst_file_path)}",
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

