from prii import (
     prii
)
from get_flattened_local_file_path import (
     get_flattened_local_file_path
)


def test_get_flattened_local_file_path_1():
    
    
    clip_id = "brewcub"
    rip_width = 1024
    rip_height = 256

    frame_indices = [
        23094,
        88369
    ]
    for frame_index in frame_indices:
        for board_id in ["left", "right"]:
            for kind in ["original", "onscreen"]:
                file_path = get_flattened_local_file_path(
                    kind=kind,
                    clip_id=clip_id,
                    frame_index=frame_index,
                    board_id=board_id,
                    rip_width=rip_width,
                    rip_height=rip_height,
                )
                
                prii(file_path)

       