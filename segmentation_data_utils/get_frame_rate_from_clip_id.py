def get_frame_rate_from_clip_id(
    clip_id: str,
):
    """
    Given a clip_id, return the frame_rate
    """

    clip_id_to_frame_rate = {  # alphabetical order to avoid duplicates
        "rwanda-2025-05-17-sdi8": 50.00,
    }
    if clip_id in clip_id_to_frame_rate:
        frame_rate = clip_id_to_frame_rate[clip_id]
    else:
        frame_rate = 59.94

    return frame_rate
