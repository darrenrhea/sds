
from typing import Tuple


def get_clip_id_and_frame_index_from_file_name(
        file_name: str
) -> Tuple[str, int]:
    assert isinstance(file_name, str), f"Invalid file_name: {file_name}"
    assert file_name.endswith('.png') or file_name.endswith('.jpg'), f"Invalid file_name: {file_name}"

    k = file_name.find('_')
    assert k != -1, f"Invalid file_name: {file_name}"
    
    clip_id = file_name[:k]
    
    frame_index = int(
        file_name[k+1:k+7]
    )

    valid_clip_ids = [
        "bos-ind-2024-01-30-mxf",
        "bos-mia-2024-04-21-mxf",
        "cle-mem-2024-02-02-mxf",
        "dal-bos-2024-01-22-mxf",
        "dal-lac-2024-05-03-mxf",
        "dal-min-2023-12-14-mxf",
        "bos-dal-2024-06-06-srt",
        "dal-bos-2024-06-11-calibration",
    ]
    assert clip_id in valid_clip_ids, f"Invalid clip_id: {clip_id}"
    prefix = f"{clip_id}_{frame_index:06d}"
    
    assert file_name.startswith(prefix), f"ERROR: {file_name} does not start with {prefix}???"

    return clip_id, frame_index
