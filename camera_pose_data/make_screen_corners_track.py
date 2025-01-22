from get_screen_corners_from_clip_id_and_frame_index import (
     get_screen_corners_from_clip_id_and_frame_index
)
from dump_as_jsonlines import (
     dump_as_jsonlines
)

from pathlib import Path


def make_a_single_file_screen_corners_track():
    clip_id = "brewcub"

    out_path = Path(f"~/{clip_id}_screen_corners_track.jsonl").expanduser()

    lst = []
    for frame_index in range(23094, 300835 + 1):
        corners = get_screen_corners_from_clip_id_and_frame_index(
            clip_id=clip_id,
            frame_index=frame_index
        )
        corners["frame_index"] = frame_index
        lst.append(corners)
    
    dump_as_jsonlines(fp=out_path, obj=lst)

    print(f"done writing {out_path=}")


if __name__ == "__main__":
    make_a_single_file_screen_corners_track()