import subprocess
import os
from pathlib import Path

generate_masks = 1
push_to_github = 0

game_name = "gsw1"
out_dir = Path('~/r/final_gsw1/led_segmentation').expanduser()
mask_name = "inbounds"
# executable_name = "mask_floor"
executable_name = "mask_surfaces"
full_executable_path = Path(f"~/r/floor-ripper/build/bin/{executable_name}.exe").expanduser()
config_path = Path(f"./gsw1_led_screens_config.json").expanduser()
file_extension = "png"

# mask_surfaces.exe config.json ~/awecom/data/clips/gsw1/tracking_attempts/unified/gsw1_585978_camera_parameters.json ./output.png

if generate_masks:
    for frame_file in os.listdir(out_dir):
        mask_number = frame_file.split("_")[1]
        print(f"mask number is {mask_number}")
        relevance_mask_path = Path(os.path.join(out_dir, game_name + "_" + mask_number + "_" + mask_name + "." + file_extension)).expanduser()
        print(f"relevance mask path is {relevance_mask_path}")
        if frame_file.endswith("nonfloor.png") and not relevance_mask_path.exists():
            floor_mask_path = Path(f"{out_dir}/gsw1_{mask_number}_{mask_name}.png").expanduser()
            camera_parameters_path = Path(f"~/awecom/data/clips/gsw1/tracking_attempts/unified/gsw1_{mask_number}_camera_parameters.json").expanduser()
            # print(f"processing {os.path.join(out_dir, frame_file)}")
            print(f"processing {mask_number}")
            print(f"{' '.join([str(full_executable_path), str(config_path), str(camera_parameters_path),str(floor_mask_path)])}")
            subprocess.run([
                str(full_executable_path),
                str(config_path),
                str(camera_parameters_path),
                str(floor_mask_path)
                ])

if push_to_github:
    for mask_number in mask_frames:
        subprocess.run(["git", "add", f"{out_dir}/gsw1_{mask_number}_{mask_name}.png"], cwd=out_dir)

    subprocess.run(["git", "commit", "-m", "relevance masks"], cwd=out_dir)
    subprocess.run(["git", "push"], cwd=out_dir)
