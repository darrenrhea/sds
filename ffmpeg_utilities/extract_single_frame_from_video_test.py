from pridiff import (
     pridiff
)
from print_metric_distances_between_hwc_np_u8s import (
     print_metric_distances_between_hwc_np_u8s
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
import numpy as np
from prii import (
     prii
)
from extract_single_frame_from_video import (
     extract_single_frame_from_video
)
from get_a_temp_file_path import (
     get_a_temp_file_path
)

from pathlib import Path


def test_extract_single_frame_from_video_jpg_version():
    """
    This tests that is correctly extracts the correct canonical frame indexed frame
    against the gold standared o a yadif mode 1 blown out video, namely:

    clip_id=munich2024-01-09-1080i-yadif

    time /usr/local/bin/ffmpeg \
    -i /shared/s3/awecomai-test-videos/nba/Mathieu/EB_23-24_R20_BAY-RMB.mxf \
    -vsync 0 \
    -vf yadif=mode=1 \
    -start_number 0 \
    /media/drhea/muchspace/clips/${clip_id}/frames/${clip_id}_%06d_original.jpg

    exiftool /media/drhea/muchspace/clips/munich2024-01-09-1080i-yadif/frames/munich2024-01-09-1080i-yadif_123456_original.jpg

    exiftool /media/drhea/muchspace/clips/munich2024-01-09-1080i-yadif/frames/munich2024-01-09-1080i-yadif_123456_original.png
    """
    
    # we are testing the jpg code path:
    png_or_jpg = "jpg"

    pix_fmt = "yuvj422p"

   
    
    input_video_abs_file_path = Path(
        "/shared/s3/awecomai-test-videos/nba/Mathieu/EB_23-24_R20_BAY-RMB.mxf"
    ).expanduser()

    fps = 50.0

    deinterlace = True

    out_frame_abs_file_path = get_a_temp_file_path(
        suffix=".jpg"
    )

    frame_index_to_extract = np.random.randint(
        low=4,
        high=300000
    )

    print(f"frame_index_to_extract={frame_index_to_extract}")

    extract_single_frame_from_video(
        input_video_abs_file_path=input_video_abs_file_path,
        frame_index_to_extract=frame_index_to_extract,
        out_frame_abs_file_path=out_frame_abs_file_path,
        fps=fps,
        deinterlace=deinterlace,
        pix_fmt=pix_fmt,
        png_or_jpg=png_or_jpg,
        verbose=False,
    )

    prii(out_frame_abs_file_path)

    should_be_abs_file_path = Path(
        f"/media/drhea/muchspace/clips/munich2024-01-09-1080i-yadif/frames/munich2024-01-09-1080i-yadif_{frame_index_to_extract:06d}_original.jpg"
    ).expanduser()
    prii(should_be_abs_file_path)

    actual = open_as_rgb_hwc_np_u8(
        image_path=out_frame_abs_file_path,
    )

    should_be = open_as_rgb_hwc_np_u8(
        image_path=should_be_abs_file_path,
    )

    print_metric_distances_between_hwc_np_u8s(
        a=actual,
        b=should_be
    )

    pridiff(
        path1=out_frame_abs_file_path,
        path2=should_be_abs_file_path
    )

   

if __name__ == "__main__":
    test_extract_single_frame_from_video_jpg_version()
    print("Tests passed")