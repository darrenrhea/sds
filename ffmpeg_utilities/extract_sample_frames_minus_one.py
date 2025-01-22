import subprocess
from pathlib import Path
import sys

usage_message = """
This script extracts every thousandth frame of an mxf video the NBA sent us,
and is intended to be executed on plumbus, a Mac machine in Houston near the videos.
This is important because the videos are too huge to move around easily.
Every thousandth frame is fairly small by comparison,
and we rsync the frames to Austin to start annotation for segmentation.

Usage:

    python extract_sample_frames.py <video_id/clip_id> <first_frame_index> <last_frame_index>
    
where:

    video_id/clip_id is something like: "SACvGSW_PGM_core_nbc_11-13-2022"

such that 

/Volumes/NBA/2022-2023_Season_Videos/<video_id>.mxf

exists.  Almost any Python 3 will work.
"""

if len(sys.argv) < 4:
    print(usage_message)
    exit(1)

try:
    first_frame_index = int(sys.argv[2])
    last_frame_index = int(sys.argv[3])
except:
    print(usage_message)
    exit(1)


video_id = sys.argv[1]

video_dir = Path("/Volumes/NBA/2022-2023_Season_Videos")
assert video_dir.is_dir(), f"ERROR: {video_dir} does not exist!"

mxf_video_base_name = f"{video_id}.mxf"

mxf_video_path = video_dir / mxf_video_base_name
assert mxf_video_path.exists(), f"ERROR: {mxf_video_path} does not exist!"

# path to Shaowei's frame extraction bash script, vid_extract:
vid_extract_path = Path(
    "/Users/awecom/src/tracker/python/utilities/scripts/vid_extract"
)
assert vid_extract_path.exists(), f"ERROR: {vid_extract_path} does not exist!"

output_dir = Path(f"~/frame_samples/{video_id}").expanduser()
output_dir.mkdir(exist_ok=True, parents=True)


for start in range(286000, 286000 + 1, 1000):
    for offset in range(0, 250+1):
        frame_index = start - offset
        frame_output_path = output_dir / f"{video_id}_{frame_index:06d}.jpg"

        print(
            f"Extracting frame_index {frame_index} from {mxf_video_base_name} to {frame_output_path}"
        )

        command_pieces = [
            str(vid_extract_path),
            str(frame_index),
            str(mxf_video_path),
            str(frame_output_path),
        ]

        completed_process = subprocess.run(args=command_pieces)

        if not frame_output_path.exists():
            print(
                f"Frame extraction failed, probably because the video:\n    "
                f"{mxf_video_base_name}\nis too short to have "
                f"a frame of index {frame_index}, "
                f"so we are going to stop the frame extraction process."
            )
            sys.exit(0)

print(
    f"\n\nNow we suggest that you do this on your local/laptop to download the extracted frames:"
)

# due to rsync's weird semantics, its probably best to make the directory exist first:
print(f"mkdir -p ~/frame_samples")

# last forward slashes are important here:
print(f"rsync -rP plumbus:~/frame_samples/ ~/frame_samples/")
