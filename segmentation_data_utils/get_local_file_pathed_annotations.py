from dvfam_denormalize_video_frame_annotations_metadata import (
     dvfam_denormalize_video_frame_annotations_metadata
)
from add_local_file_paths_to_annotation import (
     add_local_file_paths_to_annotation
)
from typing import List, Set, Tuple
# from prii_nonlinear_f32 import (
#      prii_nonlinear_f32
# )
# from load_16bit_grayscale_png_file_as_hw_np_f32 import (
#      load_16bit_grayscale_png_file_as_hw_np_f32
# )
from prii import (
     prii
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
from colorama import Fore, Style
import better_json as bj
import numpy as np
import pprint


def get_local_file_pathed_annotations(
    video_frame_annotations_metadata: List[dict],
    desired_labels: Set[str],
    desired_leagues: Set[str],  # nba, euroleague, london, etc.
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

    get_local_file_pathed_annotations_test.py
    """

    possible_desired_labels = set([
        "camera_pose",
        "depth_map",
        "floor_not_floor",
        "original",
    ])
    assert (
        set(desired_labels).issubset(possible_desired_labels)
    ), f"{desired_labels=} is mentions unknown labels.  Available labels are {possible_desired_labels=}"

    video_frame_annotations_metadata = (
        dvfam_denormalize_video_frame_annotations_metadata(
            video_frame_annotations_metadata
        )
    )

    # Especially when you are developing,
    # you may want to limit the number of annotations you are working with to certain bad / interesting ones.
    # BEGIN filter by desired_clip_id_frame_index_pairs:
    if desired_clip_id_frame_index_pairs is None:
        filtered_by_clip_id_frame_index_pairs = video_frame_annotations_metadata
    else:
        assert (
            isinstance(desired_clip_id_frame_index_pairs, list)
        ), f"Error: {desired_clip_id_frame_index_pairs=} is not a list"
        for clip_id, frame_index in desired_clip_id_frame_index_pairs:
            assert isinstance(clip_id, str), f"Error: {clip_id=} is not a string"
            assert isinstance(frame_index, int), f"Error: {frame_index=} is not an int"

        filtered_by_clip_id_frame_index_pairs = []
        for annotation in video_frame_annotations_metadata:
            clip_id = annotation["clip_id"]
            frame_index = annotation["frame_index"]
            if (clip_id, frame_index) in desired_clip_id_frame_index_pairs:
                filtered_by_clip_id_frame_index_pairs.append(annotation)
    # ENDOF filter by desired_clip_id_frame_index_pairs.
    
    # BEGIN filter by league:
    league_filtered = []
    for annotation in filtered_by_clip_id_frame_index_pairs:
        clip_id_info = annotation["clip_id_info"]
        league = clip_id_info["league"]
        if league in desired_leagues:
            league_filtered.append(annotation)
    
    # BEGIN eliminate annotations that do not have all the desired labels:
    annotations_with_at_least_the_desired_labels = []
    for annotation in league_filtered:
        clip_id = annotation["clip_id"]
        frame_index = annotation["frame_index"]
        label_name_to_sha256 = annotation["label_name_to_sha256"]

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
        if has_all_desired_labels:
            annotations_with_at_least_the_desired_labels.append(annotation)
    # ENDOF eliminate annotations that do not have all the desired labels.
    

    # BEGIN add local_file_paths for the desired_labels to the annotations:
    # Because people add all sorts of labels, you might not want to download all of the labels.
    file_pathed_annotations = [
        add_local_file_paths_to_annotation(
            annotation=annotation,
            desired_labels=desired_labels,
        )
        for annotation in annotations_with_at_least_the_desired_labels
    ]
    # ENDOF add local_file_paths for the desired_labels to the annotations.


    # Sometimes labels are corrupt, for instance a camera_pose with a focal length <= 0,
    # or which isn't even well-formed json,
    # or a png that cannot be opened.  Now that we have the label files locally, we
    # can check for corrupt data and eliminate the annotations that have corrupt data.
    # BEGIN eliminate annotations that have corruption amongst the desired labels:
    uncorrupt_file_pathed_annotations = []
    for annotation in file_pathed_annotations:
        clip_id = annotation["clip_id"]
        frame_index = annotation["frame_index"]
        local_file_paths = annotation["local_file_paths"]
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
                    print(f"focal_length is 0 in file {file_path}")
                    skip_for_corrupt_data = True
        if skip_for_corrupt_data:
            print(f"skipping {clip_id=} {frame_index=} because it has corrupt data")
        else:
            uncorrupt_file_pathed_annotations.append(annotation)
    # ENDOF eliminate annotations that have corruption amongst the desired labels.



    # Let's shuffle the annotations so that we can get a random subsample:
    np.random.shuffle(uncorrupt_file_pathed_annotations)


    if max_num_annotations is not None and len(file_pathed_annotations) > max_num_annotations:
        uncorrupt_file_pathed_annotations = uncorrupt_file_pathed_annotations[:max_num_annotations]

    if print_in_iterm2:
        for annotation in uncorrupt_file_pathed_annotations:
            pprint.pprint(annotation)
            clip_id = annotation["clip_id"]
            frame_index = annotation["frame_index"]
            label_name_to_sha256 = annotation["label_name_to_sha256"]
            clip_id_info = annotation["clip_id_info"]     
            for label_name in desired_labels:
                file_path = local_file_paths[label_name]
                assert file_path.is_file()
                prii(file_path) 
                # depth_hw_np_f32 = load_16bit_grayscale_png_file_as_hw_np_f32(depth_map_path)
                # prii_nonlinear_f32(depth_hw_np_f32)

    

    num_training_points = len(file_pathed_annotations)
    print(f"{Fore.YELLOW}{num_training_points=}{Style.RESET_ALL}")

    return uncorrupt_file_pathed_annotations

