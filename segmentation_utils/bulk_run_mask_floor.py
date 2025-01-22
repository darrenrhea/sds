import subprocess
import os
from pathlib import Path

generate_masks = 1
push_to_github = 0

out_dir = Path('~/r/final_gsw1/nonfloor_segmentation').expanduser()
mask_name = "inbounds"
executable_name = "mask_floor"
# executable_name = "mask_surfaces"
executable_path = Path(f"~/r/floor-ripper").expanduser()
full_executable_path = Path(f"~/r/floor-ripper/{executable_name}.exe").expanduser()

mask_frames = [
    # "582534",
    # "582538",
    # "585197",
    "585978"
]

# mask_surfaces.exe config.json ~/awecom/data/clips/gsw1/tracking_attempts/unified/gsw1_585978_camera_parameters.json ./output.png

if generate_masks:
    # for frame_file in os.listdir(f"{Path('~/r/gsw1/segmentation').expanduser()}"):
    #     mask_number = frame_file.split("_")[1]
    #     if frame_file.endswith("nonfloor.png"):
    for mask_number in mask_frames:       
        camera_parameters_path = Path(f"~/awecom/data/clips/gsw1/tracking_attempts/blend_second_half/gsw1_{mask_number}_camera_parameters.json").expanduser()
        floor_mask_path = Path(f"{out_dir}/gsw1_{mask_number}_{mask_name}.png").expanduser()
        # print(f"processing {os.path.join(out_dir, frame_file)}")
        print(f"processing {mask_number}")
        print(f"{' '.join([str(executable_path),str(camera_parameters_path),str(1920),str(1080),str(-47),str(47),str(-25),str(25),str(floor_mask_path)])}")
        subprocess.run([
            str(full_executable_path),
            str(camera_parameters_path),
            str(1920),
            str(1080),
            str(-47),
            # str(50)
            str(47),
            str(-25),
            str(25),
            str(floor_mask_path)
            ])
    # for frame_file in os.listdir(f"{Path('~/r/final_gsw1/final_segmentation').expanduser()}"):
    #     mask_number = frame_file.split("_")[1]
    #     if not mask_number in mask_frames:
    #         os.remove(Path(f"{out_dir}/gsw1_{mask_number}_{mask_name}.png").expanduser())

#     for mask_number in mask_frames:
#         subprocess.run(["python", "get_lane_mask.py", f"segmentation/gsw1_{mask_number}_color.png", f"~/awecom/data/clips/gsw1/tracking_attempts/second/gsw1_{mask_number}_camera_parameters.json", f"segmentation/gsw1_{mask_number}_relevant_lane.png"])

if push_to_github:
    for mask_number in mask_frames:
        subprocess.run(["git", "add", f"{out_dir}/gsw1_{mask_number}_{mask_name}.png"], cwd=out_dir)

    subprocess.run(["git", "commit", "-m", "relevance masks"], cwd=out_dir)
    subprocess.run(["git", "push"], cwd=out_dir)
