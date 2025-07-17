"""
stage_preannotations.py
makes a frame_ranges
"""
import sys
import better_json as bj
import pprint as pp
from pathlib import Path

bad_frames_json_file_path = Path(
    "~/r/frame_attributes/hou-lac-2023-11-14/hou-lac_seg_failures.json5"
).expanduser()
assert bad_frames_json_file_path.is_file(), f"bad_frames_json_file_path {bad_frames_json_file_path} is not a file"

bad_frames_jsonable = bj.load(bad_frames_json_file_path)
pp.pprint(bad_frames_jsonable)

frames = [
    x["frame"]
    for x in bad_frames_jsonable
]
pp.pprint(frames)

frame_ranges_jsonable = {
    "input_dir": "/media/drhea/muchspace/clips/hou-lac-2023-11-14/frames",
    "clip_id": "hou-lac-2023-11-14",
    "frame_ranges": frames
}

out_file_path = Path(
    "~/r/major_rewrite/preannotation_frame_ranges.json5"
).expanduser()

bj.dump(
    obj=frame_ranges_jsonable,
    fp=out_file_path
)

print(f"bat {out_file_path}")
