from extract_single_frame_from_video import extract_single_frame_from_video

from pathlib import Path


input_video_abs_file_path = Path(
    "/Volumes/awecom/baseball/10_18_2021/20200725PIT-STL-NC.mp4"
)

clip_id = "STLvPIT_2020-07-25"

out_dir = Path(
    f"~/baseball_frame_samples/{clip_id}"
).expanduser()

out_dir.mkdir(exist_ok=True, parents=True)

for frame_index_to_extract in range(0, 999999, 1000):
    out_frame_abs_file_path = out_dir / f"{clip_id}_{frame_index_to_extract:06d}.jpg"

    extract_single_frame_from_video(
        input_video_abs_file_path=input_video_abs_file_path,
        frame_index_to_extract=frame_index_to_extract,
        out_frame_abs_file_path=out_frame_abs_file_path
    )

