import pprint
import sys
from asccfn_a_simple_camera_classifier_for_nba import (
     asccfn_a_simple_camera_classifier_for_nba
)
from get_camera_pose_from_sha256 import (
     get_camera_pose_from_sha256
)
from pathlib import Path
from get_valid_camera_names import (
     get_valid_camera_names
)
from CameraParameters import (
     CameraParameters
)
from is_sha256 import (
     is_sha256
)
from ltjfthts_load_the_jsonlike_file_that_has_this_sha256 import (
     ltjfthts_load_the_jsonlike_file_that_has_this_sha256
)
from get_file_path_of_sha256 import (
     get_file_path_of_sha256
)
import copy
import better_json as bj
 

def arcpttsd_add_raguls_camera_poses_to_the_segmentation_data():
    """
    This destroys weird leagues, leaving only the nba league.
    """
    sha256_of_raguls_answers = (
        "49132e1a700146719b66c37ac32802ab4a10c9194bdcd44f56f5f399d888bbca"
    )

    sha256_of_all_segmentation_annotations = (
        "40d074a13b7ea71c9d04c6e33ba69e2c15d45b0476d79a96d3e313586fce9c3c"
    )
    
    non_nba_counter = 0

    valid_camera_names = get_valid_camera_names()
    
    raguls_answers = (
        ltjfthts_load_the_jsonlike_file_that_has_this_sha256(
            sha256_of_raguls_answers
        )
    )

    all_segmentation_annotations = (
        ltjfthts_load_the_jsonlike_file_that_has_this_sha256(
            sha256_of_all_segmentation_annotations
        )
    )
    cntr_of_camera_poses_in_all_segmentation_annotations = 0
    for x in all_segmentation_annotations:
        pprint.pprint(x)
        if x["label_name_to_sha256"]["camera_pose"] is not None:
            cntr_of_camera_poses_in_all_segmentation_annotations += 1
    print(f"{cntr_of_camera_poses_in_all_segmentation_annotations=}")

    # BEGIN CHECK raguls_answers:
    assert isinstance(raguls_answers, list)
    for record in raguls_answers:
        assert isinstance(record, dict)
        assert "original" in record
        assert "court" in record
        assert "camera_pose_candidates" in record
        original = record["original"]
        assert is_sha256(original)
        camera_pose_candidates = record["camera_pose_candidates"]
        assert isinstance(camera_pose_candidates, list)
        assert len(camera_pose_candidates) > 0

        camera_pose = camera_pose_candidates[0]
        original_sha256 = record["original"]
        original_file_path = get_file_path_of_sha256(
            sha256=original_sha256,
            check=True
        )
    # ENDOF CHECK raguls_answers.

    # BEGIN index raguls_answers by original_sha256:
    original_sha256_mapsto_raguls_answers = dict()
    for record in raguls_answers:
        original_sha256 = record["original"]
        original_sha256_mapsto_raguls_answers[original_sha256] = record
    # ENDOF index raguls_answers by original_sha256.
    

    valid_leagues = ["nba", "euroleague", "london"]
    better_segmentation_annotations = []
    for annotation in all_segmentation_annotations:
        mycopy = copy.deepcopy(annotation)
        clip_id = annotation["clip_id"]
        frame_index = annotation["frame_index"]
        label_name_to_sha256 = annotation["label_name_to_sha256"]
        original_sha256 = label_name_to_sha256["original"]
        camera_pose_sha256 = label_name_to_sha256["camera_pose"]
        if camera_pose_sha256 is None:
            chazs_camera_pose = None
        else:
            chazs_camera_pose = get_camera_pose_from_sha256(
                camera_pose_sha256
            )
        clip_id_info = annotation["clip_id_info"]
        league = clip_id_info["league"]
        assert (
            league in valid_leagues
        ), f"league is {league}, which is not in {valid_leagues=}"
        if league != "nba":
            non_nba_counter += 1
            continue

        raguls_answer = original_sha256_mapsto_raguls_answers.get(original_sha256)
        if raguls_answer is None:
            camera_name = None
            if chazs_camera_pose is not None and chazs_camera_pose.f > 0:
                camera_pose = chazs_camera_pose
        else:
            camera_pose_candidates = raguls_answer["camera_pose_candidates"]
            camera_pose_jsonable = camera_pose_candidates[0]
            # pprint.pprint(camera_pose_jsonable)
            camera_name = camera_pose_jsonable["name"]
            assert camera_name in valid_camera_names, f"{camera_name=} is not in {valid_camera_names=}"
        
            camera_pose = CameraParameters(
                rod=camera_pose_jsonable["rod"],
                loc=camera_pose_jsonable["loc"],
                f=camera_pose_jsonable["f"],
                ppi=0.0,
                ppj=0.0,
                k1=camera_pose_jsonable["k1"],
                k2=camera_pose_jsonable["k2"],
            )

       
        if camera_name is not None:
            x, y, z = camera_pose.loc
            prediction = asccfn_a_simple_camera_classifier_for_nba(
                x=x,
                y=y,
                z=z,
            )
            assert (
                prediction == camera_name
            ), f"{prediction=} {camera_name=} {x=} {y=} {z=}"
        
        if camera_name is None and camera_pose is not None:  # classifier can add the camera_name
            x, y, z = camera_pose.loc
            prediction = asccfn_a_simple_camera_classifier_for_nba(
                x=x,
                y=y,
                z=z,
            )
            camera_name = prediction

        #del mycopy["label_name_to_sha256"]["camera_pose"]
        if camera_pose is not None:
            mycopy["camera_pose"] = camera_pose.to_jsonable()
        else:
            mycopy["camera_pose"] = None
        mycopy["camera_name"] = camera_name
        better_segmentation_annotations.append(mycopy)
    
    print(f"{non_nba_counter=}")
    return better_segmentation_annotations


if __name__ == "__main__":
    better_segmentation_annotations = (
        arcpttsd_add_raguls_camera_poses_to_the_segmentation_data()
    )
    camera_pose_counter = 0
    for x in better_segmentation_annotations:
        camera_pose = x["camera_pose"]
        original_sha256 = x["label_name_to_sha256"]["original"]
        mask_sha256 = x["label_name_to_sha256"]["floor_not_floor"]
        if camera_pose is None:
            pass
            # original = get_file_path_of_sha256(sha256=original_sha256)
            # mask = get_file_path_of_sha256(sha256=mask_sha256)
            # prii(original)
            # prii(mask)
        else:
            camera_pose_counter += 1
    
    out_path = Path("~/better_segmentation_annotations.json5").expanduser()
    bj.dump(
        obj=better_segmentation_annotations,
        fp=out_path
    )
    print(f"There are {len(better_segmentation_annotations)} records in the better_segmentation_annotations")
    print(f"There are {camera_pose_counter} records with a camera_pose")
    print(f"cat {out_path}")
    print("bye")