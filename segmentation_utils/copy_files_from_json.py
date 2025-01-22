import subprocess
from pathlib import Path
import sys
import os
import json


from_dir = sys.argv[1]
to_dir = sys.argv[2]
json_path = sys.argv[3]
json_file = open(Path(json_path).expanduser())
frames_json = json.load(json_file)
print(frames_json)

# for frame_number in convert_frames:
#     from_path = Path(f"{from_dir}/{video_name}_{frame_number}.jpg").expanduser()
#     to_path = Path(f"{to_dir}/{video_name}_{frame_number}_color.png").expanduser()
#     subprocess.call(["cp", f"{from_path}", f"{to_path}"])