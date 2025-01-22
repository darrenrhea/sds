from typing import List, Set, Tuple
from prii_nonlinear_f32 import (
     prii_nonlinear_f32
)
from load_16bit_grayscale_png_file_as_hw_np_f32 import (
     load_16bit_grayscale_png_file_as_hw_np_f32
)
from prii import (
     prii
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from colorama import Fore, Style
import better_json as bj
import numpy as np


def get_local_file_paths_for_annotations(
    video_frame_annotations_metadata_sha256: str,
    desired_labels: Set[str],
    max_num_annotations: int = None,
    print_in_iterm2: bool = False,
    desired_clip_id_frame_index_pairs: List[Tuple[str, int]] = None,
    print_inadequate_annotations: bool = False
):
    """
    We have a lot of annotated video frames.
    # Originally these were segmentation annotations, but it time the need for camera_pose arose,
    as well as other things like depth maps, inset-ness, inset-masks, score_bug masks.

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

    get_local_file_paths_for_annotations_test.py
    """

    possible_desired_labels = set([
        "camera_pose",
        "depth_map",
        "floor_not_floor",
        "original"
    ])
    assert (
        set(desired_labels).issubset(possible_desired_labels)
    ), f"{desired_labels=} is mentions unknown labels.  Available labels are {possible_desired_labels=}"

    
    json_file_path = get_file_path_of_sha256(
        sha256=video_frame_annotations_metadata_sha256
    )
    segmentation_annotations = bj.load(
        json_file_path
    )

    # Let's shuffle the annotations so that we can get a random subsample:
    np.random.shuffle(segmentation_annotations)



    file_pathed_annotations = []
    for annotation in segmentation_annotations:
        if max_num_annotations is not None and len(file_pathed_annotations) >= max_num_annotations:
            break

        clip_id = annotation["clip_id"]
        frame_index = annotation["frame_index"]
        label_name_to_sha256 = annotation["label_name_to_sha256"]

        if desired_clip_id_frame_index_pairs is not None:
            if (clip_id, frame_index) not in desired_clip_id_frame_index_pairs:
                continue

        clip_id_info = annotation["clip_id_info"]
        league = clip_id_info["league"]
        if league != "nba":
            continue

        has_all_desired_labels = True
        missing_labels = set()
        for label_name in desired_labels:
            sha256 = label_name_to_sha256.get(label_name)
            if sha256 is None:
                has_all_desired_labels = False
                missing_labels.add(label_name)
        
        if print_inadequate_annotations and not has_all_desired_labels:
            print(f"skipping {clip_id=} {frame_index=} because it is missing {list(missing_labels)}")
            original_sha256 = label_name_to_sha256.get("original")
            original_file_path = get_file_path_of_sha256(original_sha256)
            prii(original_file_path)
        
        if not has_all_desired_labels:
            continue

        local_file_paths = dict()
        for label_name in possible_desired_labels:
            maybe_sha256 = label_name_to_sha256.get(label_name)
            if maybe_sha256 is not None:
                local_file_paths[label_name] = get_file_path_of_sha256(maybe_sha256)
            else:
                local_file_paths[label_name] = None
        
        for label_name in desired_labels:
            assert (
                local_file_paths.get(label_name) is not None
            ), f"{label_name=} is not in {local_file_paths=}"
        
        skip_for_corrupt_data = False
        for label_name in desired_labels:
            file_path = local_file_paths[label_name]
            assert file_path.is_file()
            if label_name == "camera_pose":
                obj = bj.load(file_path)
                assert isinstance(obj, dict)
                assert "f" in obj
                focal_length = obj["f"]
                if focal_length == 0:
                    # print(f"focal_length is 0 in file {file_path}")
                    skip_for_corrupt_data = True

        if skip_for_corrupt_data:
            continue

        if print_in_iterm2:
            for label_name in desired_labels:
                file_path = local_file_paths[label_name]
                assert file_path.is_file()
                prii(file_path)
           
            # depth_hw_np_f32 = load_16bit_grayscale_png_file_as_hw_np_f32(depth_map_path)
            # prii_nonlinear_f32(depth_hw_np_f32)
    
        annotation["local_file_paths"] = local_file_paths
        file_pathed_annotations.append(
            annotation
        )
        

    num_training_points = len(file_pathed_annotations)
    print(f"{Fore.YELLOW}{num_training_points=}{Style.RESET_ALL}")

    return file_pathed_annotations

