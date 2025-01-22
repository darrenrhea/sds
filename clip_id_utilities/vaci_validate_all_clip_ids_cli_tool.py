from validate_a_clip_entity import (
     validate_a_clip_entity
)

from load_all_clips import (
     load_all_clips
)

def vaci_validate_all_clip_ids_cli_tool():
    tuples_of_clip_id_file_path_and_clip = load_all_clips()
    for clip_id, file_path, clip_info in tuples_of_clip_id_file_path_and_clip:
        validate_a_clip_entity(
            clip_id=clip_id,
            file_path=file_path,
            clip_info=clip_info,
        )