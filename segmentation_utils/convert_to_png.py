import subprocess
from pathlib import Path

# for x in $(seq 151016 155541) ; do echo $x ; convert fastai/gsw1_${x}_nonfloor.png -alpha extract fastai_bw/gsw1_${x}_nonfloor.png ; done

# subdir = "nets1progfeed"
subdir = "BKN_CITY_2021-11-03_PGM_short"
original_dir = Path(f'~/awecom/data/clips/{subdir}/frames').expanduser()
# original_dir = Path(f'~/awecom/data/clips/{subdir}_multi/frames').expanduser()
mask_dir = Path(f'~/awecom/data/clips/gsw1/masking_attempts/final_bw').expanduser()
# segmentation_dir = Path('~/r/final_gsw1/in_progress').expanduser()
# segmentation_dir = Path('~/r/final_gsw1/led_segmentation').expanduser()
# segmentation_dir = Path(f'~/{subdir}/color').expanduser()
# segmentation_dir = Path('~/r/final_gsw1/final_segmentation').expanduser()
segmentation_dir = Path(f'~/r/brooklyn_nets_barclays_center/assignments03182022').expanduser()

convert_frames = [
    # "161000",
    # "170000",
    # "171000",
    # "173000",
    # "194000",
    # "196000",
    # "218000",
    # "221000",
    # "222000",
    # "225000",
    # "226000",
    # "227000",
    # "231000",
    # "233000",
    # "240000",
    # "241000",
    # "242000",
    # "282000",
    # "283000",
    # "288000",
    # "295000",
    # "308000",
    # "398000",
    # "415000",
    # "423000",
    # "427000",
    # "443000"
    # "002392",
    # "002400",
    # "002433",
    # "002521",
    # "004852",
    # "005328",
    # "006533",
    # "006554",
    # "006563",
    # "007200"
    "004853"

]

# filters enhance sharpness
# colors brightness contrast
begin_frame = 0
end_frame = 2000
# for index in range(begin_frame, end_frame + 1, 500):
for index in convert_frames:
    subprocess.call(["convert", "-alpha", "on", f"{original_dir}/{subdir}_{int(index):06d}.jpg", f"{segmentation_dir}/{subdir}_{int(index):06d}_color.png"])
    # subprocess.call(["convert", "-alpha", "on", f"{original_dir}/{subdir}_multi_{int(index) - 4:06d}.jpg", f"{segmentation_dir}/{subdir}_{int(index):06d}_color.png"])
    # subprocess.call(["python", "black_and_white_mask_to_rgba.py", f"{mask_dir}/gsw1_{index}_nonfloor.png", f"{segmentation_dir}/gsw1_{index}_color.png", f"{segmentation_dir}/gsw1_{index}_nonfloor.png"])

# for frame in convert_frames:
#     subprocess.call(["git", "add", f"/home/drhea/r/gsw1/segmentation/gsw1_{frame}_color.png"], cwd=Path(f'/home/drhea/r/gsw1/segmentation/').expanduser())
#     subprocess.call(["git", "commit", "-m", "color frames"], cwd=Path(f'/home/drhea/r/gsw1/segmentation/').expanduser())
#     subprocess.call(["git", "push"], cwd=Path(f'/home/drhea/r/gsw1/segmentation/').expanduser())

# for frame in convert_frames:
#     subprocess.call(["cp", f"/home/drhea/awecom/data/clips/gsw1/masking_attempts/fastai/gsw1_{frame}_nonfloor.png", f"/home/drhea/r/gsw1/segmentation/gsw1_{frame}_nonfloor.png"])

# for frame in convert_frames:
#     subprocess.call(["git", "add", f"/home/drhea/r/final_gsw1/in_progress/gsw1_{frame}_nonfloor_fastai.png"], cwd=Path(f'/home/drhea/r/final_gsw1/').expanduser())
#     subprocess.call(["git", "add", f"/home/drhea/r/final_gsw1/in_progress/gsw1_{frame}_color.png"], cwd=Path(f'/home/drhea/r/final_gsw1/').expanduser())
#     subprocess.call(["git", "commit", "-m", "nonfloor frames"], cwd=Path(f'/home/drhea/r/final_gsw1/').expanduser())
#     subprocess.call(["git", "push"], cwd=Path(f'/home/drhea/r/final_gsw1/').expanduser())
