from pathlib import Path
import argparse
import textwrap
from extract_single_frame_from_video import extract_single_frame_from_video

def eenffv_extract_every_nth_frame_from_video_cli_tool():
        
    usage_message = textwrap.dedent(
        f"""\
        This script extracts every thousandth frame of a video.

        Usage:


            mkdir -p ~/bay-mta-2024-03-22-mxf-ddv3

            eenffv_extract_every_nth_frame_from_video \\
            -n 10000 \\
            --input_video /Volumes/NBA/Euroleague/EB_23-24_R31_BAY-MTA_ddv3_deinterlaced.mov \\
            --clip_id bay-mta-2024-03-22-mxf-ddv3 \\
            --first_frame_index 0 \\
            --last_frame_index 999999 \\
            --output_dir ~/bay-mta-2024-03-22-mxf-ddv3 \\
            --fps 50.0
            
        where:

            clip_id is something like: "SACvGSW_PGM_core_nbc_11-13-2022"

        such that 

        exists.
        """
    )

    argp = argparse.ArgumentParser(usage=usage_message)
    argp.add_argument(
        "-i",
        "--input_video",
        type=str,
        help="the path to the input video file",
        required=True,
    )                               
    argp.add_argument(
        "-c",
        "--clip_id",
        type=str,
        required=True,
        help="something like: SACvGSW_PGM_core_nbc_11-13-2022 that will be the prefix of every frame extracted, then an underscore, then 6 digits of frame index, then .jpg"
    )
    argp.add_argument(
        "-n",
        "--n",
        type=int,
        required=True,
        help="extract every nth frame"
    )
    argp.add_argument("-f", "--first_frame_index", type=int, default=0, help="the first frame index to extract, starting at 0")
    argp.add_argument("-l", "--last_frame_index", type=int, default=999999, help="the last frame index to extract, starting at 0")
    argp.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="the directory to write the extracted frames to"
    )
    argp.add_argument("--fps", type=float, default=50.0, help="the frames per second of the video")
    opt = argp.parse_args()

    n = opt.n
    first_frame_index = opt.first_frame_index
    last_frame_index = opt.last_frame_index
    clip_id = opt.clip_id
    input_video = opt.input_video
    input_video_abs_file_path = Path(input_video).resolve()
    output_dir = Path(opt.output_dir).resolve()
    fps = opt.fps

    assert (
        output_dir.is_dir()
    ), f"ERROR: {output_dir} is not an extant directory!  This program does not make directories. You will have to do it."


    assert input_video_abs_file_path.exists(), f"ERROR: {input_video_abs_file_path} does not exist!"



    print(f"{first_frame_index=}, {n=}, {last_frame_index=}, {clip_id=}, {input_video_abs_file_path=}, {output_dir=}, {fps=}")
    for frame_index in range(first_frame_index, last_frame_index + 1, n):
        out_frame_abs_file_path = output_dir / f"{clip_id}_{frame_index:06d}_original.jpg"

        print(
            f"Extracting frame_index {frame_index} to {out_frame_abs_file_path}"
        )

        extract_single_frame_from_video(
            input_video_abs_file_path=input_video_abs_file_path,
            frame_index_to_extract=frame_index,
            deinterlace=False,  # TODO: make CLI flag
            fps=fps,
            pix_fmt="yuvj422p",
            png_or_jpg=out_frame_abs_file_path.suffix[1:],
            out_frame_abs_file_path=out_frame_abs_file_path
        )
    

    print(
        f"\n\nNow we suggest that you do this on your local/laptop to download the extracted frames:"
    )

    # due to rsync's weird semantics, its probably best to make the directory exist first:
    print(f"mkdir -p ~/euro")

    # last forward slashes are important here:
    print(f"rsync -rP lam:{output_dir}/ ~/euro/")

