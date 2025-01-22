import subprocess
from pathlib import Path

def extract_all_frames_from_video_as_png(
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
   
    args = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        str(input_video_abs_file_path),
        "-f",
        "image2",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(out_dir_abs_path / ffmpeg_printf_template )
    ]
    print(" \\\n".join(args))
    subprocess.run(args=args)
   
