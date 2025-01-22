import subprocess
from pathlib import Path
import sys

video_name = "curry"
original_dir = Path(f'~/awecom/data/clips/{video_name}/frames').expanduser()
mask_dir = Path(f'~/awecom/data/clips/{video_name}/masking_attempts/{video_name}').expanduser()
color_dir = Path(f'~/r/segmentation_utils/{video_name}_color').expanduser()

frame = 170519
frame_begin = int(sys.argv[1])
frame_end = int(sys.argv[2])
for index in range(frame_begin, frame_end + 1, 1):
    frame_index = f'{index:06d}'
    original_video = Path(f"{original_dir}/{video_name}_{frame_index}.jpg").expanduser()
    if original_video.is_file():
        subprocess.call(["convert", "-alpha", "on", original_video, f"{color_dir}/{video_name}_{frame_index}_color.png"])
        subprocess.call(["python", "black_and_white_mask_to_rgba.py", f"{mask_dir}/{video_name}_{frame_index}_nonfloor.png", f"{color_dir}/{video_name}_{frame_index}_color.png", f"{color_dir}/{video_name}_{frame_index}_nonfloor.png"])

