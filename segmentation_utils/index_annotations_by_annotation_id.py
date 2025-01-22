from pathlib import Path
import better_json as bj


def get_annotations_indexed_by_annotation_id():
    current_segmentation_annotations_path = Path(
        "~/r/segmentation_utils/current_segmentation_annotations.json"
    ).expanduser()

    current_segmentation_annotations = bj.load(current_segmentation_annotations_path)
    assert isinstance(current_segmentation_annotations, list)

    annotation_id_to_row = dict()
    
    for row in current_segmentation_annotations:
        clip_id = row["clip_id"]
        frame_index = row["frame_index"]
        annotation_id = f"{clip_id}_{frame_index:06d}"
        annotation_id_to_row[annotation_id] = row

    return annotation_id_to_row

