from validate_one_camera_pose import (
     validate_one_camera_pose
)
from prii import (
     prii
)
from get_original_frame_from_clip_id_and_frame_index import (
     get_original_frame_from_clip_id_and_frame_index
)
from get_camera_pose_from_clip_id_and_frame_index import get_camera_pose_from_clip_id_and_frame_index

clip_id_frame_index_pairs =[
    # ("slgame1", 150000),
    # ("slday2game1", 100000),
    # ("slday3game1", 150000),
    # ("slday4game1", 150000),
    # ("slday5game1", 150000),
    # ("slday6game1", 150000),
    ("slday7game1", 9999),
    # ("slday8game1", 150000),
    # ("slday9game1", 150000),
    # ("slday10game1", 150000),
]
for clip_id, frame_index in clip_id_frame_index_pairs:
    camera_pose = get_camera_pose_from_clip_id_and_frame_index(clip_id, frame_index)
    original_rgb_hwc_np_u8 = get_original_frame_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    
    print(f"{clip_id=}, {frame_index=}")
    prii(original_rgb_hwc_np_u8)
    drawn_on = validate_one_camera_pose(
        clip_id=clip_id,
        frame_index=frame_index,
    )
    prii(drawn_on)