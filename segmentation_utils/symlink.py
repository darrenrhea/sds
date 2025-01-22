"""
Wow. This is bad.

"""
import os
import sys
from pathlib import Path
import shutil

segmentation_model_id = "bkn_400p_400p_res34_1e_3f_crazybake_full_res"
input_dir = Path("~/r/brooklyn_nets_barclays_center/above_cameras/").expanduser()

temp_dir = Path("/awecom/data/clips/temp/frames")
temp_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path("~/after_1f/").expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

file_names = sorted(
    [
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix == ".jpg"
    ]
)

print(file_names)
assert len(file_names) == 1208


if sys.argv[1] == "forward":
        
    for index, abs_input_file_path in enumerate(file_names):
        src = abs_input_file_path
        dst = temp_dir / f"temp_{index:06d}.jpg"
        print(f"ln -s {src} {dst}")
        os.symlink(src, dst, target_is_directory=False)


if sys.argv[1] == "back":
    for index, file_name in enumerate(file_names):
        src = os.path.expanduser(f"/awecom/data/clips/temp/masking_attempts/{segmentation_model_id}/temp_{index:06d}_nonfloor.png")
        dst = output_dir / f"{file_name.stem}_nonfloor.png"
        print(f"cp {src} {dst}")
        shutil.copy(src, dst, follow_symlinks=True)

sys.exit(1)
