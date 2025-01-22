from get_screen_corners_from_clip_id_and_frame_index import (
     get_screen_corners_from_clip_id_and_frame_index
)


def gwklfcafi_get_well_known_landmarks_from_clip_id_and_frame_index(
    clip_id: str,
    frame_index: int,
):
    screen_name_to_corner_name_to_xy = (
        get_screen_corners_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
    )

    landmark_name_to_xy = dict()
    for screen_name, corner_name_to_xy in screen_name_to_corner_name_to_xy.items():
        for corner_name, xy in corner_name_to_xy.items():
            landmark_name_to_xy[f"{screen_name}_screens_{corner_name}"] = xy

    return landmark_name_to_xy
