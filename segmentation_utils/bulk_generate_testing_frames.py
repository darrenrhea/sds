import subprocess
import os
from pathlib import Path, PurePath
import sys

def is_uuid(file_string):
    """
    This is an example uuid: 7eb66122-a6a9-45d4-8c62-d39e78ac55fd
    """
    try:
        split_string = file_string.split("-")
        if len(split_string) == 5:
            if len(split_string[0]) == 8 \
                and len(split_string[1]) == 4 \
                and len(split_string[2]) == 4 \
                and len(split_string[3]) == 4 \
                and len(split_string[4]) == 12:
                    return True
        else:
            return False
    except:
        return False

# print(f"{is_uuid('augmented')}")

def video_to_frames(video_dir, frame_samples_path, video_id):
    print(f"from video_to_frames {video_dir} {frame_samples_path} {video_id}")
    args = [
        "time",
        "/usr/local/bin/ffmpeg",
        "-y",
        "-nostdin",
        "-i",
        f"{video_dir}",
        "-vsync",
        "0",
        "-q:v",
        "2",
        "-start_number",
        "0",
        f"{frame_samples_path / video_id}/{video_id}_%06d.jpg"
    ]
    subprocess.run(args)

# frame_samples_path = Path("/mnt/nas/volume1/videos/frame_samples").expanduser()
# for outer_dir in Path("/mnt/nas/volume1/videos/202304_highlights_demo").iterdir():
#     draw_id = PurePath(outer_dir).name
#     # print(f"{draw_id}")
#     if outer_dir.is_dir():
#         for video_dir in Path(outer_dir).iterdir():
#             if video_dir.is_file():
#                 video_id = Path(PurePath(video_dir).name).stem
#                 print(f"video id {video_id}")
#                 if draw_id != video_id:
#                     print(f"draw id {draw_id}")
#                     print(f"video id {video_id}")
#                     if is_uuid(video_id):
#                         (frame_samples_path / video_id).mkdir(parents=True, exist_ok=True)
#                         video_to_frames(video_dir, frame_samples_path, video_id)
#             else:
#                 # print(f"{video_dir}")
#                 for video_file in Path(video_dir).iterdir():
#                     # print(video_file)
#                     if video_file.is_file():
#                         video_id = Path(PurePath(video_file).name).stem
#                         # print(f"{video_id}")
#                     if draw_id != video_id:
#                         # print(f"video file {video_file}")
#                         print(f"draw id {draw_id}")
#                         print(f"video id {video_id}")
#                         if is_uuid(video_id):
#                             (frame_samples_path / video_id).mkdir(parents=True, exist_ok=True)
#                             video_to_frames(video_file, frame_samples_path, video_id)