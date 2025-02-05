from prii import (
     prii
)
import pprint
import sys
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
from clip_id_to_league import (
     clip_id_to_league
)

from get_enough_landmarks_to_validate_camera_pose import (
     get_enough_landmarks_to_validate_camera_pose
)
from draw_named_3d_points import (
     draw_named_3d_points
)
from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
from open_as_rgb_hwc_np_u8 import (
     open_as_rgb_hwc_np_u8
)
from get_camera_pose_from_clip_id_and_frame_index import (
     get_camera_pose_from_clip_id_and_frame_index
)

valid_camera_names = [
    "C01",
    "C02",
    "NETCAM_RIGHT",
    "NETCAM_LEFT",
    "SPIDER"
]


def vovfa_validate_one_video_frame_annotation(
    clip_id: str,
    frame_index: int,
    camera_pose: CameraParameters
):
    """
    To be well organized, we need to have a wide variety of video frames where
    we know the floor_not_floor segmentation, the camera pose,
    which floor/court it is, which teams and what uniforms they are wearing, etc.
    """

    assert isinstance(clip_id, str)
    assert isinstance(frame_index, int), f"frame_index is {frame_index}, which is not an int, but is of type {type(frame_index)}."

    league = clip_id_to_league(clip_id=clip_id)

    original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )

    original_rgb_np_u8 = open_as_rgb_hwc_np_u8(
        image_path=original_file_path
    )

    # camera_pose = get_camera_pose_from_clip_id_and_frame_index(
    #     clip_id=clip_id,
    #     frame_index=frame_index
    # ) 

    landmark_name_to_xyz = get_enough_landmarks_to_validate_camera_pose(
        league=league
    )
    print(camera_pose)
    drawn_on = draw_named_3d_points(
        original_rgb_np_u8=original_rgb_np_u8,
        camera_pose=camera_pose,
        landmark_name_to_xyz=landmark_name_to_xyz
    )

    return drawn_on

    

if __name__ == "__main__":
    
    sha256_of_raguls_answers = (
        "49132e1a700146719b66c37ac32802ab4a10c9194bdcd44f56f5f399d888bbca"
    )

    sha256_of_all_segmentation_annotations = (
        "db67948e37f4622f7ffd761c42937724a9c77efe86772988261360ad0200c156"
    )
    
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
    
    # {
    #     "clip_id": "hou-mem-2025-01-13-sdi",
    #     "frame_index": 538174,
    #     "label_name_to_sha256": {
    #         "camera_pose": null,
    #         "original": "a6dd268d4b4de9202aae94fcc53d7c3bc451e092e9c5ec1b07209e5087d301a9",
    #         "floor_not_floor": "03c0fe4b3cdb14c7e1e60ee75b7e3ab5b56c5747f5ff5c05aeb705d453393b0f"
    #     },
    #     "clip_id_info": {
    #         "youtube_url": "https://www.youtube.com/watch?v=Es0AvuWIzdQ",
    #         "home_team": "hou",
    #         "away_team": "mem",
    #         "date": "2025-01-13",
    #         "quality": "sdi",
    #         "court": "hou_core_2425",
    #         "jerseys": {
    #             "hou": "",
    #             "mem": ""
    #         },
    #         "clip_sha256": "",
    #         "old_s3_path": "",
    #         "league": "nba",
    #         "camera_poses": ""
    #     }
    # },
    valid_leagues = ["nba", "euroleague", "london"]
    for annotation in all_segmentation_annotations:
        clip_id = annotation["clip_id"]
        frame_index = annotation["frame_index"]
        label_name_to_sha256 = annotation["label_name_to_sha256"]
        original_sha256 = label_name_to_sha256["original"]
        camera_pose_sha256 = label_name_to_sha256["camera_pose"]
        clip_id_info = annotation["clip_id_info"]
        league = clip_id_info["league"]
        assert (
            league in valid_leagues
        ), f"league is {league}, which is not in {valid_leagues=}"
        if league != "nba":
            continue

        raguls_answer = original_sha256_mapsto_raguls_answers.get(original_sha256)
        if raguls_answer is None:
            print("Ragul didn't answer this one.")
            continue
        camera_pose_candidates = raguls_answer["camera_pose_candidates"]
        camera_pose_jsonable = camera_pose_candidates[0]
        # pprint.pprint(camera_pose_jsonable)
        camera_name = camera_pose_jsonable["name"]
        assert camera_name in valid_camera_names, f"{camera_name=} is not in {valid_camera_names=}"
        
        if camera_name in ["C01", "C02", "NETCAM_RIGHT", "NETCAM_LEFT"]:
            continue
    
        camera_pose = CameraParameters(
            rod=camera_pose_jsonable["rod"],
            loc=camera_pose_jsonable["loc"],
            f=camera_pose_jsonable["f"],
            ppi=0.0,
            ppj=0.0,
            k1=camera_pose_jsonable["k1"],
            k2=camera_pose_jsonable["k2"],
        )

        drawn_on = vovfa_validate_one_video_frame_annotation(
            clip_id=clip_id,
            frame_index=frame_index,
            camera_pose=camera_pose
        )

        prii(
            drawn_on,
            caption="the landmarks better line up or the camera pose is wrong:"
        )
   