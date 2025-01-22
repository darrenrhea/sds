from source_video_validator import (
     source_video_validator
)
from pathlib import Path
import sys
import textwrap
from typing import Any, Dict

from load_all_clips import (
     load_all_clips
)

def validate_a_clip_entity(
    file_path: Path,
    clip_id: str,
    clip_info: Dict[str, Any]
):
    required_top_level_keys =[
        "clip_id",
        "game_id",
        "source_video",
    ]

    required_top_level_keys_to_validator = dict(
        clip_id=None,
        game_id=None,
        source_video=source_video_validator
    )

    for key in required_top_level_keys:
        if key not in clip_info:
            print(f"{file_path!s} does not contain the {key} key")
            sys.exit(0)
    
    clip_id_from_inside = clip_info["clip_id"]
    assert (
        clip_id == clip_id_from_inside
    ), textwrap.dedent(
        f"""\
        The clip_id suggested by the immediate parent directory and file name of:
        
        {file_path}
        
        disagrees with the clip_id stated inside of that file, which says:
        
        {clip_id_from_inside=}
        """
    )
    for key, validator in required_top_level_keys_to_validator.items():
        if validator is not None:
            is_valid, reason = validator(
                info=clip_info[key]
            )
            if not is_valid:
                print(f"Problem in {file_path!s}")
                print(reason)
                sys.exit(0)
    
    source_video = clip_info["source_video"]
    if source_video["quality_level"] == "mxf":
       if "hq_start_frame_index" not in clip_info:
            print(
                textwrap.dedent(
                    f"""\
                    {file_path!s} does not contain the hq_start_frame_index key"
                    despite that the source video is of quality level mxf.
                    """
                )
            )
            sys.exit(1)


if __name__ == "__main__":
    tuples_of_clip_id_file_path_and_clip = load_all_clips()
    for clip_id, file_path, clip in tuples_of_clip_id_file_path_and_clip:
        validate_a_clip_entity(
            clip_id=clip_id,
            file_path=file_path,
            clip=clip,
        )