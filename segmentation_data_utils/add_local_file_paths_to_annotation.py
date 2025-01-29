from typing import List
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import copy


def add_local_file_paths_to_annotation(
    annotation: dict,
    desired_labels: List[str],
) -> dict:
    """
    In order that the metadata is small, as this will likely be in a database soon,
    and so we not be attached to particular storage systems (s3) or file system layouts,
    we just give the sha256s of the files that are the annotation data.
    given an video frame annotation that only mentions the hashes / sha256s,
    and a set of desired_labels, i.e. some subset of
    
    * camera_pose
    * floor_not_floor
    * original
    * depth_map
    * important_people,
    * score_bug_mask, etc.

    we want to add local file paths to the annotation and do the downloading necessary to get them.
   
    See also the test:
    
    add_local_file_paths_to_annotation_test.py
    """
    new_annotation = copy.deepcopy(annotation)
    label_name_to_sha256 = annotation["label_name_to_sha256"]
        
    local_file_paths = dict()
    for label_name in desired_labels:
        maybe_sha256 = label_name_to_sha256.get(label_name)
        if maybe_sha256 is not None:
            local_file_paths[label_name] = get_file_path_of_sha256(maybe_sha256)
        else:
            local_file_paths[label_name] = None

    new_annotation["local_file_paths"] = local_file_paths
    return new_annotation
    
