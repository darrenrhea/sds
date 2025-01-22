from pathlib import Path


def path_to_annotation_id(
    original_or_mask_file_path: Path
):
    name = original_or_mask_file_path.name
    annotation_id = None
    for suffix in ['_original.jpg', '_original.png', ".jpg", ".png"]:
        if name.endswith(suffix):
            annotation_id = name[:-len(suffix)]
            break
    assert annotation_id is not None, f"We don't know how to get the annotation_id from {original_or_mask_file_path}"
    
    return annotation_id
   