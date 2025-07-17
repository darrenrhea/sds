from make_evaluation_video import (
     make_evaluation_video
)
from pathlib import Path
import better_json as bj

# clip_id = "bay-efs-2023-12-20-mxf-yadif"
# clip_id = "bay-zal-2024-03-15-mxf-yadif"

model_id = "ana"

# ad = "skweek"
# model_id = "skw"

shared_dir = Path("/shared")


clip_id = "bay-zal-2024-03-15-mxf-yadif"
model_id = "skw"

ranges_dir = Path("~/r/frame_attributes").expanduser()
json_name = f"ad_to_ranges_for_{clip_id}.json5"
ranges_json_path = ranges_dir / json_name
print(ranges_json_path)

ad_to_ranges = bj.load(ranges_json_path)

# ranges_dir = Path("~/r/frame_attributes").expanduser()
# json_name = ranges_dir/f"ad_ranges_{clip_id}.json5"
# ranges_json_path = ranges_dir / json_name
# ad_json = bj.load(ranges_json_path)
# show_n_tell_path = Path(shared_dir).expanduser() / "show_n_tell"
# show_n_tell_clips_path = show_n_tell_path / f"{model_id}"
# show_n_tell_clips_path.mkdir(exist_ok=True)

original_suffix = "_original.jpg"
frames_dir = shared_dir / "clips" / clip_id / "frames"
masks_dir = shared_dir / "inferences"
fps = 50.0

combinations = [
    ["foreground", "green"],
]
ranges = [
    [93000, 103000]
]

show_n_tell_dir =shared_dir / "show_n_tell"

# out_suffix = ad
out_suffix = ""

for first_frame_index, last_frame_index in ranges:
    for what_is_normal_color, fill_color in combinations:

        out_video_file_path = show_n_tell_dir  / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}_{model_id}_{what_is_normal_color}_fill{fill_color}{out_suffix}.mp4"

        make_evaluation_video(
            original_suffix=original_suffix,
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            clip_id=clip_id,
            model_id=model_id,
            fps=fps,
            what_is_normal_color=what_is_normal_color,
            fill_color=fill_color,
            out_video_file_path=out_video_file_path
        )
