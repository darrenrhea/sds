import textwrap
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from extract_single_frame_from_video import (
     extract_single_frame_from_video
)

from pathlib import Path


import sys


if __name__ == "__main__":

    pairs_of_video_path_and_clip_id = [
        # already did this one: 
        # ("EB_23-24_R20_BAY-RMB.mxf", "munich2024-01-09-1080i-yadif"),
        # ("EB_23-24_R27_BAY-CZV.mxf", "bay-czv-2024-03-01"),
        # ("EB_23_24_R15_BAY-EFS.mxf", 'bay-efs-2023-12-20'),
        ("EB_23_24_R16_ZAL-BAR.mxf", "zal-bar-2023-12-22-mxf"),
    ]

    shared_dir = get_the_large_capacity_shared_directory()

    for video, clip_id in pairs_of_video_path_and_clip_id:

        input_video_abs_file_path = Path("/Volumes/NBA/Euroleague") / video
        png_or_jpg = "jpg"
        pix_fmt = "yuvj422p"
        # this is fields-per-second, not frames-per-second.
        # So if it's "25 frames per second interlaced", value should be 50 fields per second.
        fps = 50.0  
        
        frame_indices = [
            x for x in range(500, 600000, 500)
        ]
        
        out_dir = shared_dir / "clips" / clip_id / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        pairs_of_frame_index_and_abs_out_path = [
            (
                frame_index,
                out_dir / f"{clip_id}_{frame_index:06d}_original.jpg"
            )
            for frame_index in frame_indices
        ]

        for frame_index_to_extract, out_frame_abs_file_path in pairs_of_frame_index_and_abs_out_path:
            extract_single_frame_from_video(
                input_video_abs_file_path=input_video_abs_file_path,
                fps=fps,
                deinterlace=True,
                frame_index_to_extract=frame_index_to_extract,
                png_or_jpg=png_or_jpg,
                pix_fmt=pix_fmt,
                out_frame_abs_file_path=out_frame_abs_file_path,
            )
    
    for video, clip_id in pairs_of_video_path_and_clip_id:
        print(
            textwrap.dedent(
                f"""\
                Suggest that FOR A LAPTOP like aang or korra you do:

                mkdir -p ~/a/clips/{clip_id}/frames

                rsync -rP 'squanchy:~/a/clips/{clip_id}/frames/' ~/a/clips/{clip_id}/frames/

                # Then open the folder and select some frames for cutouts:
                open ~/a/clips/{clip_id}/frames/

                Suggest that for LAM you do:

                mkdir -p /media/drhea/muchspace/clips/{clip_id}/frames

                rsync -rP 'squanchy:~/a/clips/{clip_id}/frames/' /media/drhea/muchspace/clips/{clip_id}/frames/
                """
            )
    )