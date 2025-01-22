from pathlib import Path
from typing import Any, Dict, Optional
import better_json as bj


def get_clip_info_from_clip_id(
    clip_id: str
) -> Optional[Dict[str, Any]]:

    """
    Load info for the given clip_id.
    """
    clip_id_repo_dir = Path("~/r/clip_ids").expanduser()
    clips_dir = clip_id_repo_dir / "clips"
    clip_info_file_path = clips_dir / clip_id / f"{clip_id}.json5"
    print(f"Trying to open {clip_info_file_path=!s}")
    if not clip_info_file_path.exists():
        return None  
    clip_info = bj.load(clip_info_file_path)
    return clip_info


if __name__ == "__main__":
    get_clip_info_from_clip_id(
        clip_id="bay-zal-2024-03-15-mxf-yadif"
    )