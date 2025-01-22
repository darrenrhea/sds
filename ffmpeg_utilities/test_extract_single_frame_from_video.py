import subprocess
from pathlib import Path
import numpy as np
import argparse

from extract_single_frame_from_video import (
     extract_single_frame_from_video
)

def test_extract_single_frame_from_video():

    computer_name = get_what_computer_this_is()
    if computer_name == "squanchy":
        input_video_abs_file_path = Path("/Volumes/NBA/2022-2023_Season_Videos/BOSvLAC_PGM_core_bal_12-29-2022.mxf")
    
        frame_index_to_extract = 200000
        
        out_frame_abs_file_path = Path("temp.jpg")

        extract_single_frame_from_video(
            input_video_abs_file_path,
            frame_index_to_extract,
            out_frame_abs_file_path
        )

if __name__ == "__main__":
    test_extract_single_frame_from_video()