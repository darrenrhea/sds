import subprocess
from pathlib import Path
import sys
import pprint as pp
import json

usage_message = """
We need, for each video frame that we trained on, a few preceding frames from
the same video.

Usage:

    python extract_prior_frames_for_noise_model.py
"""

with open("training_frames.json", "rb") as fp:
    training_frames = json.load(fp)

# training_frames = training_frames[:20]

pp.pprint(training_frames)
# sys.exit(1)

# all videos are here:
video_dir = Path("/Volumes/NBA/2022-2023_Season_Videos")

# output_dir = Path(f"~/frame_samples/{video_id}").expanduser()
output_dir = Path(f"~/frames_for_noise_model").expanduser()
output_dir.mkdir(exist_ok=True, parents=True)

# path to Shaowei's frame extraction bash script, vid_extract:
vid_extract_path = Path(
    "/Users/awecom/src/tracker/python/utilities/scripts/vid_extract_segment"
)
assert vid_extract_path.exists(), f"ERROR: {vid_extract_path} does not exist!"

for video_id, last_frame_index in training_frames:
    assert video_dir.is_dir(), f"ERROR: {video_dir} does not exist!"
    
    mxf_video_base_name = f"{video_id}.mxf"

    mxf_video_path = video_dir / mxf_video_base_name
    if not mxf_video_path.exists():
        print(f"ERROR: {mxf_video_path} does not exist!")
        continue
 
    first_frame_index = last_frame_index - 60

    frame_output_path_template = str(output_dir) + f"/{video_id}_%06d.jpg"
    print(
        f"Extracting the frames from {first_frame_index} to {last_frame_index} from the video {mxf_video_base_name} to the format {frame_output_path_template}"
    )
    command_pieces = [
        str(vid_extract_path),
        str(first_frame_index),
        str(last_frame_index),
        str(mxf_video_path),
        str(frame_output_path_template),
    ]

    completed_process = subprocess.run(args=command_pieces)

    for frame_index in range(first_frame_index, last_frame_index + 1):
        frame_output_path = output_dir / f"{video_id}_{frame_index:06d}.jpg"
        if not frame_output_path.exists():
            print(f"ERROR: Frame extraction failed to create {frame_output_path}")
            sys.exit(1)

