from print_green import (
     print_green
)
from add_local_file_paths_to_annotation import (
     add_local_file_paths_to_annotation
)
import pprint


def test_add_local_file_paths_to_annotation_1():

    annotation = dict(
        clip_id="SL_2022_00",
        frame_index=131850,
        label_name_to_sha256=dict(
            camera_pose="5aa20685b3fec5b3c2428a61f2940429547f02cabd22dc8e30a484dc25f8e342",
            depth_map="c1b6814da97241c82430769458ef12d6d2b7b5197fcd40b83f734ac3932a6dce",
            floor_not_floor="87e59a1c14a733417a56d68478e5039a49c270b4e8e0faadb570658adb648d2d",
            original="ed557e976ea2b402386b0a6b7a8fdd9537d07885d42b7a76271a5713508fcda8",
        ),
        weird_stowaway="this is a stowaway",
    )

    desired_labels = ["camera_pose", "floor_not_floor", "original", "depth_map"]
    added = add_local_file_paths_to_annotation(
        annotation=annotation,
        desired_labels=desired_labels,
    )

    assert "local_file_paths" in added
    pprint.pprint(added)
    for key in annotation:
        assert key in added

    # Check that they were downloaded:
    for file_path in added["local_file_paths"].values():
        assert file_path.is_file()


if __name__ == "__main__":
    test_add_local_file_paths_to_annotation_1()
    print_green("add_local_file_paths_to_annotation.py has passed all tests")