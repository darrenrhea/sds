from extract_single_frame_from_video import (
     extract_single_frame_from_video
)
import textwrap
from pathlib import Path
import argparse


def esffv_extract_single_frame_from_video_cli_tool():

    argparser = argparse.ArgumentParser(
        description="Extract a single frame from a video given its frame index:\n",
        usage=textwrap.dedent(
            """\
           
            This extracts a particular frame from a video file.
            For example:

             Say you are given a timecode:
            01:38:09 frame 36
            
            (3600 * 1 + 38 * 60) * 59.94 + 36 = 352483
            esffv_extract_single_frame_from_video <video_file_path> <frame_index_to_extract> <out_frame_abs_file_path>"


            export c=bos-gsw-2022-06-08-mxf
            export sixdigits=352483

            mkdir -p ~/a/clips/${c}/frames

            esffv_extract_single_frame_from_video \\
            /Volumes/Synology512/basketball/NBA/New_Videos/BOSvGSW_C01_core_cln_06-08-2022_0001306238.mxf \\
            59.94 \\
            ${sixdigits} \\
            ~/a/clips/${c}/frames/${c}_${sixdigits}_original.jpg
            """
        )
    )

    argparser.add_argument(
        "input_video_abs_file_path", type=Path
    )
    argparser.add_argument(
        "fps", type=float
    )
    argparser.add_argument(
        "frame_index_to_extract", type=int
    )
    argparser.add_argument(
        "out_frame_abs_file_path", type=Path
    )

    argparser.add_argument(
        "--fps", type=float
    )
    args = argparser.parse_args()

    fps = args.fps

    input_video_abs_file_path = Path(
        args.input_video_abs_file_path
    ).expanduser()

    assert (
        input_video_abs_file_path.is_file()
    ), f"ERROR: {input_video_abs_file_path} does not exist!"
    
    frame_index_to_extract = args.frame_index_to_extract

    assert (
        frame_index_to_extract >= 0
    ), f"ERROR: {frame_index_to_extract} must be >= 0"

    out_frame_abs_file_path = Path(
        args.out_frame_abs_file_path
    ).expanduser()

    assert (
        out_frame_abs_file_path.suffix in [".png", ".jpg"]
    ), f"ERROR: {out_frame_abs_file_path} must have .png or .jpg suffix"

    assert (
        input_video_abs_file_path.suffix in [".mp4", ".mov", ".mxf"]
    ), f"ERROR: {input_video_abs_file_path} must have either .mp4 or .mov suffix"

    assert (
        out_frame_abs_file_path.resolve().parent.exists()
    ), f"ERROR: {out_frame_abs_file_path.resolve().parent} does not exist, as we do not make directories for you."
    
    print(
        textwrap.dedent(
            f"""\
            from {input_video_abs_file_path}
            extract frame {frame_index_to_extract}
            to {out_frame_abs_file_path}
            """
        )
    )

    extract_single_frame_from_video(
        input_video_abs_file_path=input_video_abs_file_path,
        frame_index_to_extract=frame_index_to_extract,
        deinterlace=False,  # TODO: make CLI flag
        fps=fps,
        pix_fmt="yuvj422p",
        png_or_jpg=out_frame_abs_file_path.suffix[1:],
        out_frame_abs_file_path=out_frame_abs_file_path
    )