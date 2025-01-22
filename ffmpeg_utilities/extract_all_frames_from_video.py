import subprocess
from pathlib import Path

from get_nonbroken_ffmpeg_file_path import (
     get_nonbroken_ffmpeg_file_path
)

def extract_all_frames_from_video(
    input_video_abs_file_path: Path,
    out_dir_abs_path: Path,
    clip_id: str,
    original_suffix: str,
    pix_fmt: str
):
    """
    Given the absolute path to a 59.94 fps video,
    the first_frame_index to extract (starting at 0)
    and the absolute path, extracts all the frames from 
    and saves the JPEGS to out_dir/{clip_id}_{frame_index:06d}.jpg
    """
    nonbroken_ffmpeg_file_path = get_nonbroken_ffmpeg_file_path()

    valid_original_suffixes = [
        "_original.jpg",
        "_original.png",
        ".jpg",
        ".png"
    ]
    assert (
        original_suffix in valid_original_suffixes
    ), f"ERROR: {original_suffix=} is not a valid original_suffix. Must be one of {valid_original_suffixes}"

    original_extension = original_suffix[-4:]
    if original_extension == ".jpg":
        print(f"{original_extension=}")
        assert pix_fmt in ["yuvj420p", "yuvj422p"]
    elif original_extension == ".png":
        assert pix_fmt in ["rgb24", "rgb48be"]
    else:
        raise ValueError(f"ERROR: {original_extension=} is not a valid original_extension. Must be one of ['.jpg', '.png']")
    
    assert out_dir_abs_path.is_dir()
    assert out_dir_abs_path.is_absolute()
    assert input_video_abs_file_path.is_file(), f"ERROR: {input_video_abs_file_path} is not a file"
   
    args = [
        str(nonbroken_ffmpeg_file_path),
        "-y",
        "-nostdin",
        "-i",
        str(input_video_abs_file_path),
        "-f",
        "image2",
        "-pix_fmt",
        pix_fmt,
        "-vsync",
        "0",
        "-q:v",
        "2",
        "-qmin",
        "2",
        "-qmax",
        "2",
        "-start_number",
        "0",
        str(out_dir_abs_path / f"{clip_id}_%06d{original_suffix}")
    ]
    print(" \\\n".join(args))
    subprocess.run(args=args)
   
