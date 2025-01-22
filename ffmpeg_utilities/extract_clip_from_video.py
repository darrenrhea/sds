import subprocess
from pathlib import Path


def extract_clip_from_video(
    input_video_abs_file_path: Path,
    first_frame: int,
    last_frame: int,
    out_video_abs_file_path: Path
):
    """
    Given the absolute path to a 59.94 fps video,
    the first_frame_index to extract (starting at 0)
    and the absolute path, extracts all the frames from 
    and saves the JPEGS to out_dir/{clip_id}_{frame_index:06d}.jpg
    """
    assert first_frame <= last_frame
    frame_count = last_frame - first_frame + 1
    assert input_video_abs_file_path.is_file()
    assert out_video_abs_file_path.parent.is_dir()

    seek_start = max(first_frame - 0.25, 0) / 60000 * 1001
    end_time = max(last_frame - 0.25, 0) / 60000 * 1001
    frame_count = last_frame - first_frame + 1
    print(f"{last_frame=}")
    print(f"{first_frame=}")
    print(f"{frame_count=}")
    args = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-accurate_seek",
        "-ss",
        str(seek_start),
        "-i",
        str(input_video_abs_file_path),
        "-frames",
        str(frame_count),
        str(out_video_abs_file_path)
    ]

    print(f"Running: ffmpeg {' '.join(args)}")

    subprocess.run(args=args)
   


if __name__ == "__main__":
    input_video_abs_file_path = Path("~/test_21-22_NBA_SUMR_C01_INSET+_5min_with_audio.mp4").expanduser()
    
    out_video_abs_file_path = Path("clip.mp4").resolve()
    
    first_frame= int(
        (3*60+33) * 59.94
    )
    last_frame = first_frame + 60
    print(f"{last_frame=}")
    extract_clip_from_video(
        input_video_abs_file_path=input_video_abs_file_path,
        out_video_abs_file_path=out_video_abs_file_path,
        first_frame=first_frame,
        last_frame=last_frame
    )
    
