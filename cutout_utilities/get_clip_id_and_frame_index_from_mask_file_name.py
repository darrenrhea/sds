
import textwrap
from typing import Tuple


def get_clip_id_and_frame_index_from_mask_file_name(
        file_name: str
) -> Tuple[str, int]:
    assert isinstance(file_name, str), f"Invalid file_name: {file_name}"
    suffix = "_nonfloor.png"
    assert file_name.endswith(suffix), f"Invalid mask name: {file_name}"
    annotation_id = file_name[:-len(suffix)]
    clip_id = annotation_id[:-7]
    assert annotation_id[-7] == "_"
    frame_index = int(annotation_id[-6:])
    assert (
        file_name == f"{clip_id}_{frame_index:06d}_nonfloor.png"
    ), textwrap.dedent(
        f"""\
        Invalid file_name:
        {file_name}
        because
        {clip_id=}
        and
        {frame_index=}
        """
    )
    return clip_id, frame_index
