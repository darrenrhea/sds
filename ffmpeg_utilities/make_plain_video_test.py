from make_plain_video import (
     make_plain_video
)
from get_mother_dir_of_frames_dir_from_clip_id import (
     get_mother_dir_of_frames_dir_from_clip_id
)
from color_print_json import (
     color_print_json
)
from load_json_file import (
     load_json_file
)
from pathlib import Path


def test_make_plain_video_1():
    obj = load_json_file(
        Path("~/r/frame_attributes/summer_league_evaluationvideos.json5").expanduser()
    )
    color_print_json(obj)

    for k in range(len(obj)):
        clip_id, first_frame_index, last_frame_index = obj[k]

        frames_dir = get_mother_dir_of_frames_dir_from_clip_id(clip_id) / "clips" / clip_id / "frames"
        original_suffix = "_original.jpg"
        out_video_file_path = Path("/shared/show_n_tell") / f"{clip_id}_from_{first_frame_index}_to_{last_frame_index}.mp4"
        make_plain_video(
            original_suffix=original_suffix,
            frames_dir=frames_dir,
            first_frame_index=first_frame_index,
            last_frame_index=last_frame_index,
            clip_id=clip_id,
            fps=59.94,
            out_video_file_path=out_video_file_path,
        )

if __name__ == "__main__":
    test_make_plain_video_1()
