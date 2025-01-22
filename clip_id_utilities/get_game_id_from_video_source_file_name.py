import textwrap
from load_all_clips import (
     load_all_clips
)
from typing import Any, Dict, Optional


def get_game_id_from_video_source_file_name(
    file_name: str
) -> Optional[Dict[str, Any]]:
    """
    This searches through all clip_ids looking for a source_video
    with the given file_name.
    """
    assert isinstance(file_name, str)
    
    print(f"Trying to get the game_id for the video called {file_name}")
    # this will fail if anyone is not JSON5:
    tuples_of_clip_id_file_path_and_clip = load_all_clips()
    
    candidates = set()
    for clip_id, file_path, clip_info in tuples_of_clip_id_file_path_and_clip:
        game_id = clip_info["game_id"]
        if clip_info["source_video"]["file_name"] == file_name:
            candidates.add(game_id)

    candidates = list(candidates)
    if len(candidates) == 0:    
        return None
    elif len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print(f"Found more than one mxf-quality clip for the game_id: {game_id}")
        print(candidates)
        raise Exception()

