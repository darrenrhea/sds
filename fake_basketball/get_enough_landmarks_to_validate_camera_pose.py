from get_nba_geometry import (
     get_nba_geometry
)

from get_euroleague_geometry import (
     get_euroleague_geometry
)

from get_led_corners import (
     get_led_corners
)


def get_enough_landmarks_to_validate_camera_pose(
    league: str
) -> dict:
    """
    This function returns enough landmarks to validate a camera pose
    for a Dallas Mavericks NBA court.
    """
    assert league == "nba"
    
    if league == "euroleague":
        geometry = get_euroleague_geometry()
    elif league == "nba":
        geometry = get_nba_geometry()

    landmark_name_to_xyz = geometry["points"]
   
    filtered_landmark_name_to_xyz = dict()
    for key in landmark_name_to_xyz.keys():
        if key.endswith("_m") or key.endswith("_oo") or "corner" in key:
            filtered_landmark_name_to_xyz[key] = landmark_name_to_xyz[key]

    dct = get_led_corners(
        court_id="ThomasMack"
    )

    for key in dct.keys():
        filtered_landmark_name_to_xyz[key] = dct[key]
    

    return filtered_landmark_name_to_xyz
