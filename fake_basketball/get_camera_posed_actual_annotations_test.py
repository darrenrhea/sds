from get_camera_posed_actual_annotations import (
     get_camera_posed_actual_annotations
)
from CameraParameters import CameraParameters

def test_get_camera_posed_actual_annotations_1():
    """
    see segmentation_data_utils/get_local_file_paths_for_annotations_test.py
    for a different way.
    """
    repo_ids_to_use = [
        "munich1080i_led",
        "bay-mta-2024-03-22-mxf_led",
    ]

    camera_posed_actual_annotations = get_camera_posed_actual_annotations(
        repo_ids_to_use=repo_ids_to_use
   )
    
    for camera_posed_actual_annotation in camera_posed_actual_annotations:
        camera_pose = camera_posed_actual_annotation["camera_pose"]
        assert isinstance(camera_pose, CameraParameters)
       
 


    
if __name__ == "__main__":
    test_get_camera_posed_actual_annotations_1()