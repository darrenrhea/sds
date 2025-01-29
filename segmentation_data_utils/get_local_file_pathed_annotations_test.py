import pprint
from get_local_file_pathed_annotations import (
     get_local_file_pathed_annotations
)
import better_json as bj


def test_get_local_file_pathed_annotations_1():
    """
    Tests retrieval and prii-ing of depth_map, floor_not_floor, and original.
    """
    desired_labels = set(["depth_map", "floor_not_floor", "original"])
    video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"
    local_file_pathed_annotations = get_local_file_pathed_annotations(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        desired_labels=desired_labels,
        desired_leagues=["nba"],
        max_num_annotations=None,
        print_in_iterm2=True,
    )
    for a in local_file_pathed_annotations:
        pprint.pprint(a)
        assert a["local_file_paths"]["original"].is_file()
        assert a["local_file_paths"]["floor_not_floor"].is_file()
        assert a["local_file_paths"]["depth_map"].is_file()


def test_get_local_file_pathed_annotations_2():
    """
    Tests retrieval and prii-ing of non_trivial_camera_pose, floor_not_floor, and original.
    """
    video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"
    desired_labels = set(["camera_pose", "floor_not_floor", "original"])
    local_file_pathed_annotations = get_local_file_pathed_annotations(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        desired_labels=desired_labels,
        desired_leagues=["nba"],
        max_num_annotations=None,
        print_in_iterm2=False,
        print_inadequate_annotations = False,
    )

    for a in local_file_pathed_annotations:
        assert a["local_file_paths"]["original"].is_file()
        assert a["local_file_paths"]["floor_not_floor"].is_file()
        assert a["local_file_paths"]["camera_pose"].is_file()
        nontrivial_camera_pose_file_path = a["local_file_paths"]["camera_pose"]
        obj = bj.load(nontrivial_camera_pose_file_path)
        assert isinstance(obj, dict)
        assert "f" in obj
        focal_length = obj["f"]
        assert focal_length > 0, f"focal_length is 0 or negative in file {nontrivial_camera_pose_file_path}"
    
    # pprint.pprint(local_file_pathed_annotations)

if __name__ == "__main__":
    test_get_local_file_pathed_annotations_1()
    test_get_local_file_pathed_annotations_2()
