from get_nonbroken_ffmpeg_file_path import (
     get_nonbroken_ffmpeg_file_path
)

from what_computer_is_this import (
     what_computer_is_this
)

from whoami import (
     whoami
)

from get_font_file_path import (
     get_font_file_path
)

from pathlib import Path
import subprocess
import textwrap


def get_valid_video_file_extensions():
    """
    TODO: Move this to a shared module.
    """
    valid_video_file_extensions = [
            ".ts",
            ".mp4",
            ".mxf",
            ".mov",
        ]
    return valid_video_file_extensions


def add_frame_numbers_to_video(
    in_video_file_path: Path,
    first_frame_index: int,
    out_video_file_path: Path,
):  
    """
    Suppose you already have video but you need it to have frame numbers on it
    and be quicktime playable. This function will do that for you.
    """
    assert in_video_file_path.is_absolute(), f"ERROR: in_video_file_path {in_video_file_path} is not an absolute path!"
    assert in_video_file_path.is_file(), f"ERROR: in_video_file_path {in_video_file_path} is not an extant file!"
    assert out_video_file_path.parent.is_dir(), f"ERROR: out_dir {out_video_file_path.parent} is not an extant directory!"
    valid_video_file_extensions = get_valid_video_file_extensions()

    assert (
        out_video_file_path.suffix in get_valid_video_file_extensions()
    ), textwrap.dedent(
        f"""\
        ERROR: out_video_file_path
        {out_video_file_path}
        does not have a a valid movie/video suffix! Possible suffixes are:
        {valid_video_file_extensions}
        """
    )

   
    font_file_path = get_font_file_path()
    ffmpeg = get_nonbroken_ffmpeg_file_path()
    
    args = [
        str(ffmpeg),
        "-y",
        "-i",
        str(in_video_file_path),
        "-vf",
        f"drawtext=fontfile={font_file_path}: text='%{{frame_num}}': start_number={first_frame_index}: x=w-tw: y=3*lh: fontcolor=yellow: fontsize=50: box=1: boxcolor=black: boxborderw=5",
        "-vsync",
        "0",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "22",
        str(out_video_file_path),
    ]

    print("  \\\n".join(args))

    subprocess.run(
        args=args
    )

    # Form the ssh alias to rsync from:
    username_computer = whoami() + what_computer_is_this()

    # instruct how to download the video(s) to your laptop:
    print(
        textwrap.dedent(
            f"""\
            On your laptop do:

                rsync -P {username_computer}:{out_video_file_path} ~/show_n_tell

                open ~/show_n_tell/{out_video_file_path.name}
            """
        )
    )

 
