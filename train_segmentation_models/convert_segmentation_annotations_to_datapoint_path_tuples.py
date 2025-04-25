from gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded import (
     gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded
)


def convert_segmentation_annotations_to_datapoint_path_tuples(
    segmentation_annotations: list,
):
    datapoint_path_tuples = []
    for x in segmentation_annotations:
        mask_sha256 = x["mask_sha256"]
        original_sha256 = x["original_sha256"]
        mask_path = gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
            sha256=mask_sha256,
            check=False,
        )
        assert mask_path is not None
        original_path = gfposaiwad_get_file_path_of_sha256_assuming_it_was_already_downloaded(
            sha256=original_sha256,
            check=False,
        )
        assert original_path is not None
        datapoint_path_tuples.append(
            (original_path, mask_path, None)
        )
    return datapoint_path_tuples

   