import subprocess
from pathlib import Path
from extract_clip_from_video import extract_clip_from_video
from concatenate_videos_together import concatenate_videos_together
from typing import List

def vaccilate():

    videos_to_concatenate = []

    temp_dir = Path("temp").resolve()
    temp_dir.mkdir(exist_ok=True, parents=True)

    video_A_path = Path("~/tracker_for_assets/registries/nba/sources/test_21-22_NBA_SUMR_C01.mp4").expanduser()
    video_B_path = Path("~/tracker_for_assets/registries/nba/sources/test_21-22_NBA_SUMR_C01_INSET+.mp4").expanduser()
    video_paths = [
        video_A_path,
        video_B_path
    ]

    for segment_index in range(5):
        for video_index, video_path in enumerate(video_paths):
            first_frame = 1800 * segment_index
            last_frame = 1800 * (segment_index + 1) - 1
            out_video_abs_file_path = temp_dir / f"{video_index}_{segment_index}.mp4"
            input_video_abs_file_path = video_path

            extract_clip_from_video(
                input_video_abs_file_path=input_video_abs_file_path,
                first_frame=first_frame,
                last_frame=last_frame,
                out_video_abs_file_path=out_video_abs_file_path
            )

            videos_to_concatenate.append(out_video_abs_file_path)
    
    
    concatenate_videos_together(
        video_abs_file_paths=videos_to_concatenate,
        out_video_abs_file_path=Path("vacillate.mp4").resolve()
    )


if __name__ == "__main__":
    vaccilate()
    
