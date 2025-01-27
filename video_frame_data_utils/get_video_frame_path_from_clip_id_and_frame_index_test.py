from get_video_frame_path_from_clip_id_and_frame_index import (
     get_video_frame_path_from_clip_id_and_frame_index
)
import textwrap

from pathlib import Path

from prii import (
     prii
)


def test_get_video_frame_path_from_clip_id_and_frame_index_1() -> None:
    clip_id = "bos-mia-2024-04-21-mxf"
    frame_index = 734500
   

    original_file_path = get_video_frame_path_from_clip_id_and_frame_index(
        clip_id=clip_id,
        frame_index=frame_index
    )
    
    assert (
        original_file_path.is_file()
    ), textwrap.dedent(
        f"""\
        ERROR:
        {original_file_path}
        is not a a local file?
        """
    )
    
    prii(original_file_path)
    return
    

if __name__ == "__main__":
    test_get_video_frame_path_from_clip_id_and_frame_index_1()
    print("get_video_frame_path_from_clip_id_and_frame_index has passed all tests.")