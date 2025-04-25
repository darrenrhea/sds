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
        
        # (
        #     "/Volumes/NBA/2022-2023_Season_Videos/CHAvIND_PGM_core_bal_11-16-2022.mxf",
        #     "cha-ind-2022-11-16-mxf",
        # ),
        # (
        #     "/Volumes/NBA/2022-2023_Season_Videos/BKNvIND_PGM_core_yes_10-31-2022.mxf",
        #     "bkn-ind-2022-10-31-mxf",
        # ),
        # (
        #     "/Volumes/NBA/2023-2024_Season_Videos/NBA_LED_Tests/MILvIND_PGM_core_tnt_04-21-2024.mxf",
        #     "mil-ind-2024-04-21-mxf",
        # ),
        # (
        #     "/Volumes/NBA/2022-2023_Season_Videos/INDvTOR_PGM_core_bal_11-12-2022.mxf",
        #     "ind-tor-2022-11-12-mxf",
        # ),
        # (
        #     "/Volumes/NBA/2022-2023_Season_Videos/BOSvDEN_PGM_core_alt_11-11-2022.mxf",
        #     "bos-den-2022-11-11-mxf",
        # ),
        # (
        #     "/Volumes/NBA/2023-2024_Season_Videos/Incoming/DALvBOS_PGM_core_esp_06-12-2024.mxf",
        #     "dal-bos-2024-06-12-mxf",
        # ),
        # (
        #     "/shared/s3/awecomai-original-videos/bos-dal-2024-06-06-unaugmented_srt_fullgame.mp4",
        #     "bos-dal-2024-06-06-srt",
        # ),
        # (
        #     "/Volumes/NBA/2023-2024_Season_Videos/Originals/BOSvDAL_PGM_core_esp_06-09-2024.mxf",
        #     "bos-dal-2024-06-09-mxf",
        # ),
        # (
        #     "/shared/s3/awecomai-original-videos/bos-dal-2024-06-06-unaugmented_srt_fullgame.mp4",
        #     "bos-dal-2024-06-06-srt",
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvLAL_LAR_city_ssn_01_05_2024.mxf",
        #     clip_id="HOUvLAL_LAR_city_ssn_01_05_2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvBOS_LAR_city_ssn_01_03_2024.mxf",
        #     clip_id="HOUvBOS_LAR_city_ssn_01_03_2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvBOS_HNT_city_ssn_01_03_2024.mxf",
        #     clip_id="HOUvBOS_HNT_city_ssn_01_03_2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvLAL_HNT_city_ssn_01_05_2024.mxf",
        #     clip_id="HOUvLAL_HNT_city_ssn_01_05_2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvLAL_RAR_city_ssn_01_05_2024.mxf",
        #     clip_id="HOUvLAL_RAR_city_ssn_01_05_2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvBOS_RAR_city_ssn_01_03_2024.mxf",
        #     clip_id="HOUvBOS_RAR_city_ssn_01_03_2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvLAC_LAR_core_ssn_11-13-2024.mxf",
        #     clip_id="HOUvLAC_LAR_core_ssn_11-13-2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvLAC_RAR_core_ssn_11-13-2024.mxf",
        #     clip_id="HOUvLAC_RAR_core_ssn_11-13-2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),
        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Originals/HOUvLAC_HNT_core_ssn_11-13-2024.mxf",
        #     clip_id="HOUvLAC_HNT_core_ssn_11-13-2024",
        #     start=140000,
        #     end=330000,
        #     step=1000,
        # ),

        # dict(
        #     video="/Volumes/NBA/2023-2024_Season_Videos/Incoming/GSWvLAC_PGM_core_nbc_10-27-2024.mxf",
        #     clip_id="GSWvLAC_PGM_core_nbc_10-27-2024",
        #     start=0,
        #     end=900000,
        #     step=1000,
        # ),

        dict(
            video="/Volumes/NBA/2023-2024_Season_Videos/Incoming/GSWvLAC_PGM_core_nbc_10-27-2024.mxf",
            clip_id="GSWvDEN_PGM_core_esp_03-17-2025",
            start=0,
            end=900000,
            step=1000,
        ),

    ]

    color_has_been_fixed = [
        (
            "/Volumes/NBA/2023-2024_Season_Videos/Originals/DALvBOS_PGM_core_bal_01-22-2024.mxf",
            "dal-bos-2024-01-22-mxf",
        ),
        (
            "/Volumes/NBA/2023-2024_Season_Videos/Originals/DALvLAC_PGM_city_esp_05-03-2024_game.mxf",
            "dal-lac-2024-05-03-mxf",
        ),
        (
            "/Volumes/NBA/2022-2023_Season_Videos/CLEvMEM_PGM_core_tnt_02-02-2023.mxf",
            "cle-mem-2024-02-02-mxf",
        ),
        (
            "/Volumes/NBA/2023-2024_Season_Videos/NBA_LED_Tests/BOSvMIA_PGM_core_esp_04-21-2024.mxf",
            "bos-mia-2024-04-21-mxf",
        ),
        (
            "/Volumes/NBA/2023-2024_Season_Videos/Originals/DALvMIN_PGM_core_bal_12-14-2023.mxf",
            "dal-min-2023-12-14-mxf",
        ),
        (
            "/Volumes/NBA/2023-2024_Season_Videos/Originals/BOSvIND_PGM_core_tnt_01-30-2024.mxf",
            "bos-ind-2024-01-30-mxf",
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
        fps = 59.94  
        
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
                print(f"ERROR: {out_frame_abs_file_path} does not exist, quitting because this usually means you have run off the end of the video")
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

                mkdir -p /hd2/clips/{clip_id}/frames

                rsync -rP 'squanchy:~/a/clips/{clip_id}/frames/' /hd2/clips/{clip_id}/frames/
                """
            )
    )