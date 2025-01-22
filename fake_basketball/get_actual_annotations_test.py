from get_actual_annotations import (
     get_actual_annotations
)

if __name__ == "__main__":
    actual_annotations = get_actual_annotations()
    print("actual_annotations:")
    for annotation in actual_annotations:
        actual_annotation_original_file_path = annotation["actual_annotation_original_file_path"]
        actual_annotation_mask_file_path = annotation["actual_annotation_mask_file_path"]
        clip_id = annotation["clip_id"]
        frame_index = annotation["frame_index"]

        
        print(f"{clip_id=}")
        print(f"{frame_index=}")
        print(f"{actual_annotation_original_file_path=!s}")
        print(f"{actual_annotation_mask_file_path=!s}")

