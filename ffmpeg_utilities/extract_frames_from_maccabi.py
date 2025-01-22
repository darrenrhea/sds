import textwrap
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)
from extract_from_this_video_these_frames import (
     extract_from_this_video_these_frames
)
from extract_single_frame_from_video import (
     extract_single_frame_from_video
)

from pathlib import Path


import sys


if __name__ == "__main__":
    shared_dir = get_the_large_capacity_shared_directory()
    print(shared_dir)

    
    # input_video_abs_file_path = Path(
    #     "/Volumes/NBA/Euroleague/EB_23-24_R02_EA7-MTA.mxf"
    # )

    # clip_id = "milan2023-10-31-1080i-yadif"
        
    # input_video_abs_file_path = Path(
    #     "/Volumes/NBA/Euroleague/EB_23-24_R13_VIR-MTA.mxf"
    # )
    # clip_id = "bologna2023-12-08-1080i-yadif"

    input_video_abs_file_path = Path(
        "/Volumes/NBA/Euroleague/EB_23-24_R21_PAR-MTA.mxf"
    )
    clip_id = "belgrade2024-01-12-1080i-yadif"

    

    png_or_jpg = "jpg"
    pix_fmt = "yuvj422p"
    fps = 50.0  # this is fields-per-second, not frames-per-second.  So if it's "25 frames per second interlaced", value should be 50 fields per second.
    
    frame_indices = [
        x for x in range(91000, 91000 + 30000 + 1, 1)
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