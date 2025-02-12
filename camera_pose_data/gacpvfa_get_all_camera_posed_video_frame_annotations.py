from pathlib import Path
from arcpttsd_add_raguls_camera_poses_to_the_segmentation_data import (
     arcpttsd_add_raguls_camera_poses_to_the_segmentation_data
)
import better_json as bj
 

def gacpvfa_get_all_camera_posed_video_frame_annotations():

    better_segmentation_annotations = (
        arcpttsd_add_raguls_camera_poses_to_the_segmentation_data()
    )
    
    all_camera_posed_video_frame_annotations = []
    for x in better_segmentation_annotations:
        camera_pose = x["camera_pose"]
        # original_sha256 = x["label_name_to_sha256"]["original"]
        # mask_sha256 = x["label_name_to_sha256"]["floor_not_floor"]
        if camera_pose is None:
            pass
            # original = get_file_path_of_sha256(sha256=original_sha256)
            # mask = get_file_path_of_sha256(sha256=mask_sha256)
            # prii(original)
            # prii(mask)
        else:
            all_camera_posed_video_frame_annotations.append(x)

    out_path = Path("~/all_camera_posed_video_frame_annotations.json5").expanduser()
    bj.dump(
        obj=all_camera_posed_video_frame_annotations,
        fp=out_path
    )
    print(f"There are {len(all_camera_posed_video_frame_annotations)} camera posed video frame annotations.")
    print(f"cat {out_path}")

    return all_camera_posed_video_frame_annotations