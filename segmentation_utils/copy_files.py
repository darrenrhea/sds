import subprocess
from pathlib import Path
import sys
import os


from_dir = sys.argv[1]
to_dir = sys.argv[2]
video_name = sys.argv[3]
# python copy_files.py \~/r/final_gsw1/nonfloor_segmentation \~/r/final_gsw1/led_segmentation
# python copy_files.py \~/r/final_gsw1/final_segmentation \~/r/final_gsw1/led_segmentation

convert_frames = [
    # "150331",
    # "150900",
    # "151382",
    # "153700",
    # "160800",
    # "161000",
    # "167200",
    # "193372"
    # "150000",
    # "150893",
    # "151037",
    # "150291",
    # "151004"
    # "148785",
    # "153700",
    # "153752",
    # "154000",
    # "163266",
    # "163576",
    # "163768",
    # "163833",
    # "163851",
    # "163973",
    # "164044",
    # "165013",
    # "165719",
    # "165723",
    # "169348",
    # "169370",
    # "170501",
    # "170506",
    # "305648",
    # "305706",
    # "328980",
    # "331545",
    # "331849",
    # "332931",
    # "333322",
    # "499171",
    # "499201",
    # "499227",
    # "499325",
    # "582534",
    # "582538",
    # "585197",
    # "585978"
    # "002392",
    # "002400",
    # "002433",
    # "002521",
    # "004852",
    # "005328",
    # "006533",
    # "006554",
    # "006563",
    # "007200",
    "004853"
]

# for frame in convert_frames:
#     subprocess.call(["git", "add", f"/home/drhea/r/gsw1/segmentation/gsw1_{frame}_color.png"], cwd=Path(f'/home/drhea/r/gsw1/segmentation/').expanduser())
#     subprocess.call(["git", "commit", "-m", "color frames"], cwd=Path(f'/home/drhea/r/gsw1/segmentation/').expanduser())
#     subprocess.call(["git", "push"], cwd=Path(f'/home/drhea/r/gsw1/segmentation/').expanduser())

for frame_number in convert_frames:
# for frame_file in os.listdir(Path(to_dir).expanduser()):
    # if frame_file.endswith("nonfloor.png"):
    # frame_number = frame_file.split("_")[1]
    from_path = Path(f"{from_dir}/{video_name}_{frame_number}.jpg").expanduser()
    to_path = Path(f"{to_dir}/{video_name}_{frame_number}_color.png").expanduser()
    subprocess.call(["cp", f"{from_path}", f"{to_path}"])
        # subprocess.call(["cp", f"{from_dir}/gsw1_{frame_number}_nonfloor.png", f"{to_dir}/gsw1_{frame_number}_nonfloor.png"])

# for frame in convert_frames:
#     # subprocess.call(["git", "add", f"{to_dir}/gsw1_{frame}_nonfloor.png"], cwd=Path(f'{to_dir}').expanduser())
#     # subprocess.call(["git", "commit", "-m", "nonfloor frames"], cwd=Path(f'{to_dir}').expanduser())
#     # subprocess.call(["git", "push"], cwd=Path(f'{to_dir}').expanduser())
#     subprocess.call(["git", "add", f"{to_dir}/gsw1_{frame}_color.png"], cwd=Path(f'{to_dir}').expanduser())
#     subprocess.call(["git", "commit", "-m", "color frames"], cwd=Path(f'{to_dir}').expanduser())
#     subprocess.call(["git", "push"], cwd=Path(f'{to_dir}').expanduser())
