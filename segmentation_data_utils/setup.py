from setuptools import setup

setup(
    name="segmentation_data_utils",
    version="0.1.0",
    py_modules=[
        "add_local_file_paths_to_annotation",
        "dvfam_denormalize_video_frame_annotations_metadata",
        "get_all_video_frame_annotations",
        "get_flattened_file_path",
        "get_flattened_local_file_path",
        "get_flattened_onscreen_mask_hw_np_u8",
        "get_flattened_original_hwc_np_u8",
        "get_human_annotated_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention",
        "get_human_annotated_mask_path_from_clip_id_and_frame_index_and_segmentation_convention",
        "get_local_file_pathed_annotations",
        "get_mask_hw_np_u8_from_clip_id_and_frame_index_and_convention_and_final_model_id",
        "get_mask_path_from_clip_id_and_frame_index_and_model_id",
        "get_mask_path_from_original_path",
        "get_original_path",
        "gpksafasf_get_primary_keyed_segmentation_annotations_from_a_single_folder",
        "maybe_find_sister_original_path_of_this_mask_path",
        "migrate_video_frame_annotations",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
        ],
    },
)
