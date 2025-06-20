import pprint
from get_enough_landmarks_to_validate_camera_pose import (
     get_enough_landmarks_to_validate_camera_pose
)


def test_get_enough_landmarks_to_validate_camera_pose_1():
    leagues = [
        # "nba",
        "nfl",
    ]
    for league in leagues:

        ans = get_enough_landmarks_to_validate_camera_pose(
            league=league
        )

        pprint.pprint(ans)

if __name__ == "__main__":
    test_get_enough_landmarks_to_validate_camera_pose_1()