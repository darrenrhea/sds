from get_nonbroken_ffmpeg_file_path import (
     get_nonbroken_ffmpeg_file_path
)
import subprocess
from pathlib import Path
from print_green import print_green

def extract_all_rgba_frames_from_video_as_png(
    input_video_abs_file_path: Path,
    out_dir_abs_path: Path,
    ffmpeg_printf_template: str = "%05d.png"
):
    """
    Intended for LED videos they gave us.
    Given the absolute path to a video,
    and the absolute path, extracts all the frames from 
    and saves the PNGs to out_dir_abs_path/{frame_index}.png
    """
    assert out_dir_abs_path.is_dir()
    assert out_dir_abs_path.is_absolute()
    nonbroken_ffmpeg_file_path = get_nonbroken_ffmpeg_file_path()

    args = [
        str(nonbroken_ffmpeg_file_path),
        "-y",
        "-nostdin",
        "-i",
        str(input_video_abs_file_path),
        "-f",
        "image2",
        "-pix_fmt",
        "rgba",  # this is what makes it RGBA
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(out_dir_abs_path / ffmpeg_printf_template )
    ]
    print_green(" \\\n".join(args))
    subprocess.run(
        args=args,
        capture_output=False,
    )
   
