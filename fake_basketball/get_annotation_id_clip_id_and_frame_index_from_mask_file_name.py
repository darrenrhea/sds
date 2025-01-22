def get_annotation_id_clip_id_and_frame_index_from_mask_file_name(
    mask_file_name: str
):
    """
    We can parse the annotation_id, clip_id, and frame_index from the mask_file_name.
    See the test.
    """
    assert mask_file_name.endswith("_nonfloor.png")
    annotation_id = mask_file_name[:-len("_nonfloor.png")]
    assert annotation_id[-7] == "_"
    clip_id = annotation_id[:-7]
    frame_index = int(annotation_id[-6:])
    assert mask_file_name == f"{clip_id}_{frame_index:06d}_nonfloor.png"
    assert mask_file_name == f"{annotation_id}_nonfloor.png"
    return annotation_id, clip_id, frame_index


