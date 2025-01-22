from colorama import Fore, Style
from pathlib import Path
import subprocess
from hash_tools import sha256_of_file


def download_sha256_from_s3(
    sha256: str,
    destination_file_path: Path,
    verbose: bool = False,
    extension: str = "",
):
    """
    You say which sha256 you want,
    and where you want to place it at.
    """
    if verbose:
        print(f"Placing file with sha256 {sha256} at:")
        print(f"{destination_file_path}")
    assert isinstance(sha256, str)
    assert len(sha256) == 64
    assert isinstance(destination_file_path, Path)
    assert destination_file_path.is_absolute()
    assert not destination_file_path.is_dir()
    assert destination_file_path.parent.is_dir(), f"ERROR: the parent directory {destination_file_path.parent} does not exist!"
    assert isinstance(verbose, bool)

    # Download the file from S3:
    completed_process = subprocess.run(
        args=[
            "aws",
            "s3",
            "cp",
            f"s3://awecomai-shared/sha256/{sha256}{extension}",
            str(destination_file_path)
        ],
        cwd=Path("~").expanduser()
    )
    if completed_process.returncode != 0:
        print("ERROR: download failed!")
        return False
    
    # Check that the sha256 is correct:
    recalculated_sha256 = sha256_of_file(destination_file_path)
    if recalculated_sha256 != sha256:
        print("ERROR: the sha256 of the downloaded file is incorrect!")
        return False
    if verbose:
        print(f"{Fore.GREEN}The sha256 checked out!{Style.RESET_ALL}")
    return True


if __name__ == "__main__":
    download_sha256_from_s3(
        sha256="c9fea27896de968772fa0193e1523364138267cacc6e743c988a1d43934979b8",
        destination_file_path=Path("dog.png").resolve(),
        verbose=False
    )