from get_all_video_frame_annotations import (
     get_all_video_frame_annotations
)


def make_map_from_clip_id_and_frame_index_to_video_frame_annotation(
    video_frame_annotations_metadata_sha256
):
    """
    Chaz has put camera-poses in json files next to the image files,
    but we prefer to look up the camera-poses from the clip_id and frame_index.
    """
        
    video_frame_annotations = get_all_video_frame_annotations(
        video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256,
        download_all_referenced_files=False,
    )

    map_from_clip_id_and_frame_index_to_video_frame_annotation = dict()

    for video_frame_annotation in video_frame_annotations:
        clip_id = video_frame_annotation["clip_id"]
        frame_index = video_frame_annotation["frame_index"]
        map_from_clip_id_and_frame_index_to_video_frame_annotation[
             (clip_id, frame_index)
        ] = video_frame_annotation

    return map_from_clip_id_and_frame_index_to_video_frame_annotation


if __name__ == "__main__":
    video_frame_annotations_metadata_sha256 = "99bc2c688a6bd35f08b873495d062604e0b954244e6bb20f5c5a76826ac53524"
    map_from_clip_id_and_frame_index_to_video_frame_annotation = (
        make_map_from_clip_id_and_frame_index_to_video_frame_annotation(
            video_frame_annotations_metadata_sha256=video_frame_annotations_metadata_sha256
        )
    )