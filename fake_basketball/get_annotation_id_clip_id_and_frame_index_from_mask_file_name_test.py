from pathlib import Path
from typing import List

from get_annotation_id_clip_id_and_frame_index_from_mask_file_name import (
     get_annotation_id_clip_id_and_frame_index_from_mask_file_name
)

def test_get_annotation_id_clip_id_and_frame_index_from_mask_file_name_1():
    mask_file_name = "bay-ber-2024-03-22-mxf_012345_nonfloor.png"
    annotation_id, clip_id, frame_index = get_annotation_id_clip_id_and_frame_index_from_mask_file_name(
        mask_file_name=mask_file_name
    )
    print(f"{annotation_id=}")
    print(f"{clip_id=}")
    print(f"{frame_index=}")
    assert annotation_id == "bay-ber-2024-03-22-mxf_012345"
    assert clip_id == "bay-ber-2024-03-22-mxf"
    assert frame_index == 12345
    print("get_annotation_id_clip_id_and_frame_index_from_mask_file_name passed")

if __name__ == "__main__":
    test_get_annotation_id_clip_id_and_frame_index_from_mask_file_name_1()