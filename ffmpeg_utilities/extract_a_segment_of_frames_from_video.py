import subprocess
from pathlib import Path


def extract_a_segment_of_frames_from_video(
    input_video_abs_file_path: Path,
    first_frame: int,
    last_frame: int,
    out_dir_abs_path: Path,
    clip_id: str
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
    seek_start = max(first_frame - 0.25, 0) / 60000 * 1001

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
        "yuvj420p",
        "-vsync",
        "0",
        "-q:v",
        "2",
        "-qmin",
        "2",
        "-qmax",
        "2",
        "-frames:v",
        str(frame_count),
        "-start_number",
        "0",
        str(out_dir_abs_path / f"{clip_id}_%06d.jpg")
    ]
    print(" \\\n".join(args))
    subprocess.run(args=args)
   


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
    
