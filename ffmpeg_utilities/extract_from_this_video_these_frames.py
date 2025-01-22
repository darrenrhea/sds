from extract_single_frame_from_video import (
     extract_single_frame_from_video)

from pathlib import Path
from typing import List, Tuple


def extract_from_this_video_these_frames(
    input_video_abs_file_path: Path,
    pairs_of_frame_index_and_abs_out_path: List[Tuple[int, Path]],
    fps: float,
    deinterlace: bool,
    png_or_jpg: str,
    pix_fmt: str,
):
    """
    Give this the absolute path to a video file,
    and a List of pairs, where each pair is a frame index (starting at 0) paired with the
    absolute path where you want that video frame to be written-to-file
    """
    for frame_index_to_extract, out_frame_abs_file_path in pairs_of_frame_index_and_abs_out_path:
        extract_single_frame_from_video(
            input_video_abs_file_path=input_video_abs_file_path,
            frame_index_to_extract=frame_index_to_extract,
            out_frame_abs_file_path=out_frame_abs_file_path,
            deinterlace=deinterlace,
            fps=fps,
            png_or_jpg=png_or_jpg,
            pix_fmt=pix_fmt,
        )
