from typing import List
from is_sha256 import (
     is_sha256
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import better_json as bj
import numpy as np


def get_all_video_frame_annotations(
    video_frame_annotations_metadata_sha256: str,
    download_all_referenced_files: bool = True,
) -> List[dict]:
    """
    We have a lot of annotated video frames.
    Originally these were floor_not_floor segmentation annotations,
    but in time the need for camera_pose arose,
    as well as other things like depth maps, inset-ness, inset-masks, score_bug masks,
    landmarks / keypoints on the court, etc.

    some of which may be inappropriate or inapplicable for what you are trying to do.
    Say you are trying to train a floor_not_floor model,
    you would not want wood_not_wood annotations.

    Today we need annotations that have a depthmap and a floor_not_floor segmentation.
    Any annotation that does not have those "labels" is not useful to us, so we filter them out.

    You may also want to limit to certain sports or leagues, like just NBA.

    You may also want to limit to a certain number of annotations, like 1000,
    often as a fast way of testing.


    A given training attempt may need certain features
    attached to the training annotations,
    like a depth map, a floor_not_floor segmentation,
    a camera_pose, relevance masks, etc.
    See also the test,

    get_local_file_pathed_annotations_test.py
    """

    valid_label_names = set(
        [
            "camera_pose",
            "depth_map",
            "floor_not_floor",
            "original",
            "floor_not_floor_relevance_mask",
            "led",
            "led_relevance_mask",
        ]
    )
    
    
    json_file_path = get_file_path_of_sha256(
        sha256=video_frame_annotations_metadata_sha256
    )
    # TODO: make load json with line numbers function so that we can complain:
    segmentation_annotations = bj.load(
        json_file_path
    )

    for record in segmentation_annotations:
        assert isinstance(record, dict), f"{record=}"
        assert "clip_id" in record, f"{record=}"
        assert "frame_index" in record, f"{record=}"
        assert "label_name_to_sha256" in record, f"{record=}"
        label_name_to_sha256 = record["label_name_to_sha256"]
        assert isinstance(label_name_to_sha256, dict), f"{label_name_to_sha256=}"
        for label_name, file_sha256 in label_name_to_sha256.items():
            assert isinstance(label_name, str), f"{label_name=}"
            assert label_name in valid_label_names, f"{label_name=}"
            if file_sha256 is not None:
                assert is_sha256(file_sha256), f"{file_sha256=} could not be a sha256"
                if download_all_referenced_files:
                    get_file_path_of_sha256(sha256=file_sha256, check=True)

    # Let's shuffle the annotations so that we can get a random subsample:
    np.random.shuffle(segmentation_annotations)

    return segmentation_annotations


if __name__ == "__main__":
    video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"
    get_all_video_frame_annotations(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        download_all_referenced_files=True,
    )
