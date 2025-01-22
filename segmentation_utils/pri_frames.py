from pathlib import Path
import os
import math
import shutil


frames_dir = Path('/awecom/data/clips').expanduser()
Path(os.path.join(frames_dir, "inset_clips")).mkdir(parents=True, exist_ok=True)
inset_clips_dir = os.path.join(frames_dir, "inset_clips")

for folder in os.listdir(frames_dir):
    full_dir = os.path.join(frames_dir, folder)
    # print(f"full_dir {full_dir}")
    if os.path.isdir(full_dir):
        for subfolder in os.listdir(full_dir):
            # print(f"subfolder is {subfolder}")
            if subfolder == "frames":
                counter = 0
                file_path = os.path.join(full_dir, subfolder)
                # print(f"number of files {len(os.listdir(file_path))}")
                if len(os.listdir(file_path)) > 100000:
                    # print(f"{file_path}")
                    for member in os.listdir(file_path):
                        # print(f"with extension {member}")
                        if member.endswith(".jpg"):
                            without_extension = os.path.splitext(member)[0]
                            if folder in without_extension:
                                # print(f"{folder}")
                                # print(f"without extension {without_extension}")
                                frame_number = int(without_extension.split("_")[-1])
                                if frame_number % 1000 == 0 and frame_number >= 150000:
                                    # print(f"{frame_number}")
                                    shutil.copy(os.path.join(file_path, member), inset_clips_dir)

                # for member in os.listdir(file_path):
                #     print(f"member {member}")
                #     if os.path.isfile(member):
                #         counter += 1
                # print(f"counter {counter}")