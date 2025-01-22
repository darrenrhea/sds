from pathlib import Path
from typing import Any, Dict, List, Tuple
import better_json as bj


def load_all_clips() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Load all clips.
    Will complain if the directory name and the file name do not match.
    """
    clip_id_repo_dir = Path("~/r/clip_ids").expanduser()
    
    clips_dir = clip_id_repo_dir / "clips"
    tuples_of_clip_id_file_path_and_clip = []
    for p in clips_dir.iterdir():
        if not p.is_dir():
            continue
        clip_id_according_to_dir = p.name
        json_file_path = p / f"{clip_id_according_to_dir}.json5"
        assert (
            json_file_path.exists()
        ), f"{json_file_path=} does not exist:{json_file_path}"
        
        clip = bj.load(json_file_path)

        tuples_of_clip_id_file_path_and_clip.append(
            (
                clip_id_according_to_dir,
                json_file_path,
                clip,
            )
        )
    return tuples_of_clip_id_file_path_and_clip


if __name__ == "__main__":
    load_all_clips()