from load_all_clips import (
     load_all_clips
)
from typing import Any, Dict, Optional


def maybe_get_mxf_clip_id_from_game_id(
    game_id: str
) -> Optional[Dict[str, Any]]:
    """
    A given game may have multiple videos of it from
    youtube, srt capture, Mathieu filming, etc.
    We want the canonically frame-indexed mxf blowout.
    Currently if the mxf is interlaced this means yadif.
    """

    """
    Load info for the given clip_id.
    """
    
    assert isinstance(game_id, str), f"{game_id=}"
    print(f"Trying to get the (a?) mxf-quality clip_id for the game_id: {game_id}")
    # this will fail if anyone is not JSON5:
    tuples_of_clip_id_file_path_and_clip = load_all_clips()
    
    num_candidates = 0
    for clip_id, file_path, clip_info in tuples_of_clip_id_file_path_and_clip:
        if clip_info["game_id"] == game_id:
            print(f"Found a clip_id for the game_id: {clip_id}")
            if clip_info["source_video"]["quality_level"] == "mxf":
                candidate = clip_id
                num_candidates += 1

    if num_candidates == 0:    
        return None
    elif num_candidates == 1:
        return candidate
    elif num_candidates > 1:
        print(f"Found more than one mxf-quality clip for the game_id: {game_id}")
        raise Exception()
    return candidate


if __name__ == "__main__":
    game_id = "bay-zal-2024-03-15"
    game_id = "bay-zal-2024-03-15"

    clip_id = maybe_get_mxf_clip_id_from_game_id(
        game_id=game_id
    )
    print(f"For {game_id=}, the mxf quality clip_id is {clip_id=}")