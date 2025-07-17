import cv2
import numpy as np
from pathlib import Path
import shutil

# dataset_folder = Path(
#     "~/fakemunich"
# ).expanduser()

# out_folder = Path(
#     "~/fakemunich_clahe_tgs32"
# ).expanduser()
# out_folder.mkdir(exist_ok=True)

dataset_folder = Path(
    "/media/drhea/muchspace/clips/unaugmented-stream-capture-2024-03-22/frames"
).expanduser()

out_folder = Path(
    "/media/drhea/muchspace/clips/unaugmented-stream-capture-2024-03-22_clahe_t32/frames"
).expanduser()
out_folder.mkdir(exist_ok=True)

add_suffix = 1

suffix = "_clahe_t32.jpg"

for member in dataset_folder.rglob("*"):
    # print(member.stem.split("_")[-1])
    frame_type = member.stem.split("_")[-1]
    if add_suffix == 1:
        out_path = out_folder/(member.stem+suffix)
    else:
        out_path = out_folder/member.name
    if frame_type == "original":
        # print(f"{frame_type=}")
        # print(f"Not copying {frame_type}")
        bgr = cv2.imread(str(member))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        bgr_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # print(f"{out_path=}")
        cv2.imwrite(str(out_path), bgr_out)
    # else:
    #     # print(f"{frame_type=}")
    #     shutil.copy(member, out_path)

# eaffv_extract_all_frames_from_video "/home/drhea/unaugmented-stream-capture-2024-03-22.mp4" unaugmented-stream-capture-2024-03-22 "/media/drhea/muchspace/clips/unaugmented-stream-capture-2024-03-22/frames/"

# export clip_id=unaugmented-stream-capture-2024-03-22
    
# python infer_cli_tool.py \
# --final_model_id c32 \
# --clip_id ${clip_id} \
# --original_suffix _original_clahe_t32.jpg \
# --start 0 \
# --end 14273 \
# --step 1

# export clip_id=unaugmented-stream-capture-2024-03-22    
# export a=0 
# export b=14273
# export m=c32
# time mev_make_evaluation_video \
# --original_suffix _original.jpg \
# --frames_dir $(shared_dir)/clips/${clip_id}/frames \
# --masks_dir $(shared_dir)/inferences \
# --clip_id ${clip_id} \
# --first_frame_index "$a" \
# --last_frame_index "$b" \
# --model_id ${m} \
# --fps 50.0

# ffmpeg -i /media/drhea/muchspace/show_n_tell/unaugmented-stream-capture-2024-03-22_from_0_to_14273_c32_foreground.mp4 -i /media/drhea/muchspace/show_n_tell/unaugmented-stream-capture-2024-03-22_from_0_to_14273_csL_foreground.mp4 -filter_complex vstack=inputs=2 /media/drhea/muchspace/show_n_tell/unaugmented-stream-capture-2024-03-22_from_0_to_14273_clahet32vnoclahe.mp4

# python infer_cli_tool.py \
# --final_model_id fmt \
# --clip_id ${clip_id} \
# --original_suffix _original.jpg \
# --start 0 \
# --end 14273 \
# --step 1