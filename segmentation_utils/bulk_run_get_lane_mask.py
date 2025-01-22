import subprocess
import os
from pathlib import Path

generate_masks = 1
push_to_github = 0

mask_frames = [
    "223731"
    # "150331",
    # "150900",
    # "151382",
    # "151681",
    # "153700",
    # "156500",
    # "158052",
    # "159089",
    # "160800",
    # "161000",
    # "167200",
    # "153100",
    # "190923",
    # "223731",
    # "224394",
    # "229041",
    # "229145",
    # "229388",
    # "229431",
    # "229562",
    # "229776",
    # "230115",
    # "231502",
    # "243284",
    # "246940"
]


if generate_masks:
    for frame_file in os.listdir("./segmentation"):
        if frame_file.endswith("nonlane.png"):
            mask_number = frame_file.split("_")[1]
            filename = Path(f"segmentation/gsw1_{mask_number}_relevant_lane.png")
            # if not filename.exists():
            print(f"processing {os.path.join('/segmentation', frame_file)}")
            subprocess.run(["python", "get_lane_mask.py", f"segmentation/gsw1_{mask_number}_color.png", f"~/awecom/data/clips/gsw1/tracking_attempts/second/gsw1_{mask_number}_camera_parameters.json", f"segmentation/gsw1_{mask_number}_relevant_lane.png"])
#     for mask_number in mask_frames:
#         subprocess.run(["python", "get_lane_mask.py", f"segmentation/gsw1_{mask_number}_color.png", f"~/awecom/data/clips/gsw1/tracking_attempts/second/gsw1_{mask_number}_camera_parameters.json", f"segmentation/gsw1_{mask_number}_relevant_lane.png"])

# if push_to_github:
#     for mask_number in mask_frames:
#         subprocess.run(["git", "add", f"segmentation/gsw1_{mask_number}_relevant_lane.png"])

#     subprocess.run(["git", "commit", "-m", "relevant lane masks"])
#     subprocess.run(["git", "push"])
