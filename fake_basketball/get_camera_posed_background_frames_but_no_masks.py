from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)
from get_the_large_capacity_shared_directory import (
     get_the_large_capacity_shared_directory
)

from CameraParameters import CameraParameters


def get_camera_posed_background_frames_but_no_masks(
    clip_id,
    original_suffix,
    first_frame_index,
    last_frame_index,
    step
):
    """
    Sometimes you need original video frames with their own camera pose,
    but you don't need masks.
    Like when you overwrite the people and other foreground objects with a floor texture and LED boards,
    which should erase the great majority of them.
    """
    # dotflat made this flat directory of good stuff from the approvals.json5:
    # approved_dir = Path("~/r/munich1080i_led/.approved").expanduser()
    shared_dir = get_the_large_capacity_shared_directory()

    approved_dir = shared_dir / "clips" / clip_id / "frames"

    approved_annotations = []
    for frame_index in range(first_frame_index, last_frame_index, step):
        annotation_id = f"{clip_id}_{frame_index:06d}"
        original_file_path = approved_dir / f"{annotation_id}{original_suffix}"
        assert original_file_path.exists(), f"{original_file_path} does not exist"

        camera_parameters = get_camera_pose_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
        assert isinstance(camera_parameters, CameraParameters)
        
        if camera_parameters.rod[0] == 0.0:
            continue

        approved_annotations.append(
            dict(  # an "annotation" this dictionary with these pieces of info:
                annotation_id=annotation_id,
                original_file_path=original_file_path,
                frame_index=frame_index,
                clip_id=clip_id,
                camera_pose=camera_parameters,
            )
        )
    
    return approved_annotations
