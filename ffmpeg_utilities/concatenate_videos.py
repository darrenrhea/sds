from get_nonbroken_ffmpeg_file_path import (
     get_nonbroken_ffmpeg_file_path
)
import subprocess
from pathlib import Path
from typing import List
import tempfile


def concatenate_videos(
    video_abs_file_paths: List[Path],
    out_video_abs_file_path: Path,
    verbose: bool = True
):
    """
    Given the absolute-file-paths of several videos,
    make a video that is the concatenation of all of them
    in that order and save it to the given output file path.
    """
    ffmpeg_executable_path = get_nonbroken_ffmpeg_file_path()

    for x in video_abs_file_paths:
        assert x.resolve().is_file(), f"ERROR: {x} is not an extant file."

    assert len(video_abs_file_paths) > 0
    assert out_video_abs_file_path.is_absolute()

    assert (
        out_video_abs_file_path.parent.is_dir()
    ), "ERROR: The parent directory of the output video file path {} must already exist, we don't make directories."
    temporary_abs_file_path = Path(
        tempfile.NamedTemporaryFile(suffix="txt", delete=False).name
    )
    assert temporary_abs_file_path.exists()
    assert temporary_abs_file_path.is_file()

    lines = [
        f"file '{video_abs_file_path}'"
        for video_abs_file_path in video_abs_file_paths
    ]

    # add the lines to the file:
    with open(temporary_abs_file_path, "w") as f:
        f.write("\n".join(lines))
    


    # ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4

    args = [
        str(ffmpeg_executable_path),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(temporary_abs_file_path),
        "-c",
        "copy",
        str(out_video_abs_file_path),
    ]

    if verbose:
        multiline_command = ' \\\n'.join(args)
        print(f"Running:\n{multiline_command}")

    subprocess.run(args=args)
   


if __name__ == "__main__":
    video_abs_file_paths = [
        Path("/mnt/data/www/f854f56d10f4d29f50c62f6b2be2bfc727c6874dc34a19118d3a291c3b6ce55a.webm").expanduser(), # ABC song
        Path("/mnt/data/www/17f65000eb31cf2475119165eaea04ea2a88c3b449b453eac33b9e65781ce50d.webm").expanduser(), # babyshark
        Path("/mnt/data/www/98f6e85aa4891c5b48f98d954230fa9fcfc2a2cffd0d09ae8a5a0006e050bf22.webm").expanduser(), # wheelonthebus
        Path("/mnt/data/www/93b1d5f849af76d2e2aee79d04f8dd70e874c72e37cd5d114cc19a73e18dfc7d.webm").expanduser(), #pasta
        Path("/mnt/data/www/7f6b087d07ad237c757da373580d96024f6a5cfc74beee64c4f726421b1a0220.webm").expanduser(), # itsybitsyspider
    ]

    out_video_abs_file_path = Path("/mnt/data/www/concat.webm").expanduser().resolve()
    
    concatenate_videos(
        video_abs_file_paths=video_abs_file_paths,
        out_video_abs_file_path=out_video_abs_file_path
    )
    
