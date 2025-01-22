from prii import (
     prii
)
from make_relevance_mask_for_led_boards import (
     make_relevance_mask_for_led_boards
)
from download_file_via_rsync import (
     download_file_via_rsync
)
from get_world_coordinate_descriptors_of_ad_placements import (
     get_world_coordinate_descriptors_of_ad_placements
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)


if __name__ == "__main__":

    # any camera_posed_human_annotation is also enough to be a camera_posed_original_video_frames:
    # camera_posed_original_video_frames = get_a_few_camera_posed_human_annotations()
    
    clip_id = "bay-zal-2024-03-15-mxf-yadif"
    
    ad_placement_descriptors = get_world_coordinate_descriptors_of_ad_placements(
        clip_id=clip_id,
        with_floor_as_giant_ad=False,
        overcover_by=0.2,
    )
    ad_placement_descriptors = [
        x
        for x in ad_placement_descriptors
        if x.name != "LEDBRD0"
    ]

    frame_index = 94960
    shared_dir = get_the_large_capacity_shared_directory()
    remote_shared_dir = get_the_large_capacity_shared_directory("lam")
    original_name = f"{clip_id}_{frame_index:06d}_original.jpg"
    local_frames_dir_path = shared_dir / "clips" / clip_id / "frames"
    local_original_file_path = local_frames_dir_path / original_name
    local_frames_dir_path.mkdir(parents=False, exist_ok=True)
    remote_original_file_path = remote_shared_dir / "clips" / clip_id / "frames" / original_name

    if not local_original_file_path.is_file():
        download_file_via_rsync(
            src_machine="lam",
            src_file_path=remote_original_file_path,
            dst_file_path=local_original_file_path,
            verbose=True
        )

    camera_pose = get_camera_pose_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    ) 

    camera_posed_original_video_frame = dict(
        original_file_path=local_original_file_path,
        frame_index=frame_index,
        clip_id=clip_id,
        camera_pose=camera_pose,
    )
    
    relevance_mask = make_relevance_mask_for_led_boards(
        camera_posed_original_video_frame=camera_posed_original_video_frame,
        ad_placement_descriptors=ad_placement_descriptors
    )

    prii(relevance_mask)
   