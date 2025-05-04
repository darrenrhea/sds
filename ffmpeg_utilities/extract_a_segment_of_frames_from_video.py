from print_red import (
     print_red
)
from print_green import (
     print_green
)
import subprocess
from pathlib import Path


def extract_a_segment_of_frames_from_video(
    input_video_abs_file_path: Path,
    first_frame: int,
    last_frame: int,
    out_dir_abs_path: Path,
    clip_id: str,
    verbose: bool,
    fps: float,
):
    """
    Given the absolute path to a 59.94 fps video,
    the first_frame_index to extract (starting at 0)
    and the absolute path, extracts all the frames from 
    and saves the JPEGS to out_dir/{clip_id}_{frame_index:06d}.jpg
    """
    assert first_frame <= last_frame
    frame_count = last_frame - first_frame + 1
    Path(out_dir_abs_path).expanduser().mkdir(parents=True, exist_ok=True)
    assert out_dir_abs_path.is_dir()
    assert out_dir_abs_path.is_absolute()
    seek_start = max(first_frame - 0.25, 0) / fps # 60000 * 1001

    args = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-accurate_seek",
        "-ss",
        str(seek_start),
        "-i",
        str(input_video_abs_file_path),
        "-f",
        "image2",
        "-pix_fmt",
        "yuvj422p",
        "-vsync",
        "0",
        "-q:v",
        "1",
        "-qmin",
        "1",
        "-qmax",
        "1",
        "-frames:v",
        str(frame_count),
        "-start_number",
        f"{first_frame}",
        str(out_dir_abs_path / f"{clip_id}_%06d.jpg")
    ]
    if verbose:
        print_green(" \\\n".join(args))
    subprocess.run(
        args=args,
        capture_output=True,
    )
    file_paths = [
        out_dir_abs_path / f"{clip_id}_{i:06d}.jpg"
        for i in range(first_frame, last_frame + 1)
    ]
    extant_file_paths = [
        file_path
        for file_path in file_paths
        if file_path.is_file()
    ]
    return extant_file_paths


if __name__ == "__main__":
    input_video_abs_file_path = Path(
        "/mnt/nas/volume1/videos/nba/Finals22/GSWvBOS_PGM_core_esp_06-02-2022_0001306207_NBA202200021086930001r.mxf"
    )
    out_dir_abs_path = Path(
        "~/frames_for_noise_model"
    ).expanduser()

    first_frame = 376500 - 60
    last_frame = 376500
    clip_id = "GSWvBOS_06-02-2022_PGM_ESP_MXF"

    extract_a_segment_of_frames_from_video(
        input_video_abs_file_path=input_video_abs_file_path,
        first_frame=first_frame,
        last_frame=last_frame,
        out_dir_abs_path=out_dir_abs_path,
        clip_id=clip_id
    )
    
