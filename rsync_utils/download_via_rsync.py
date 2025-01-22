import subprocess
from pathlib import Path


def download_via_rsync(
    src_machine,
    src_path: Path,
    dst_path: Path,
    verbose: bool = False
) -> bool:
    """
    Because the semantics of rsyncing directories is a bit complicated with respect to whether it:
    1. makes a new subdirectory named the same as the src directory's name
    2. or just moves the contents of the source directory into the destination directory
    we could do much better than this.  But for now, we'll just assume that the source path is a file.
    """
    assert src_path.is_absolute()
    assert dst_path.is_absolute(), f"{dst_path=} is not absolute.  We need an absolute path since resolving relative paths would require knowing the current working directory."

    assert src_machine in ["lam", "squanchy", "plumbus"]

    args = [
        "rsync",
        "-rP",
        f"{src_machine}:{str(src_path)}",
        str(dst_path)
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

