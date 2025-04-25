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

    pairs_of_video_path_str_and_clip_id = [
        dict(
            video="/hd2/s3/awecomai-mxf-dropbox/NFL_ISOs/59773/SkyCam.mp4",
            clip_id="nfl-59773-skycam",
            start=500,
            end=900000,
            step=1000,
        ),
        dict(
            video="/hd2/s3/awecomai-mxf-dropbox/NFL_ISOs/59778/SkyCam.mp4",
            clip_id="nfl-59778-skycam",
            start=500,
            end=900000,
            step=1000,
        ),
    ]

    

    shared_dir = get_the_large_capacity_shared_directory()

    for dct in pairs_of_video_path_str_and_clip_id:
        video_path_str = dct["video"]
        clip_id = dct["clip_id"]
        first_frame_index = dct["start"]
        last_frame_index = dct["end"]
        step = dct["step"]
        input_video_abs_file_path = Path(video_path_str)
        assert input_video_abs_file_path.exists()
        png_or_jpg = "jpg"
        pix_fmt = "yuvj422p"
        # png_or_jpg = "png"
        # pix_fmt = "rgb48be"
        # this is fields-per-second, not frames-per-second.
        # So if it's "25 frames per second interlaced", value should be 50 fields per second.
        fps = 29.97
        
        frame_indices = [
            x for x in range(first_frame_index, last_frame_index + 1, step) 
        ]
        
        out_dir = shared_dir / "clips" / clip_id / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        pairs_of_frame_index_and_abs_out_path = [
            (
                frame_index,
                out_dir / f"{clip_id}_{frame_index:06d}_original.{png_or_jpg}"
            )
            for frame_index in frame_indices
        ]

        for frame_index_to_extract, out_frame_abs_file_path in pairs_of_frame_index_and_abs_out_path:
            extract_single_frame_from_video(
                input_video_abs_file_path=input_video_abs_file_path,
                fps=fps,
                deinterlace=False,
                frame_index_to_extract=frame_index_to_extract,
                png_or_jpg=png_or_jpg,
                pix_fmt=pix_fmt,
                out_frame_abs_file_path=out_frame_abs_file_path,
            )
            if not out_frame_abs_file_path.is_file():
                print(f"ERROR: {out_frame_abs_file_path} does not exist, breaking because this usually means you have run off the end of the video")
                break
    
    for dct in pairs_of_video_path_str_and_clip_id:
        video_path_str = dct["video"]
        clip_id = dct["clip_id"]

        print(
            textwrap.dedent(
                f"""\
                Suggest that FOR A LAPTOP like aang or korra you do:

                mkdir -p ~/a/clips/{clip_id}/frames

                rsync -rP 'squanchy:~/a/clips/{clip_id}/frames/' ~/a/clips/{clip_id}/frames/

                # Then open the folder and select some frames for cutouts:
                open ~/a/clips/{clip_id}/frames/

                Suggest that for lam you do:

                mkdir -p /media/drhea/muchspace/clips/{clip_id}/frames

                rsync -rP 'squanchy:~/a/clips/{clip_id}/frames/' /shared/clips/{clip_id}/frames/
                """
            )
    )