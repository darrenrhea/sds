from setuptools import setup

setup(
    name="alpha_compositing_utils",
    version="0.1.0",
    py_modules=[
        "PasteableCutout",
        "assert_cutout_metadata_is_good",
        "augment_cutout",
        "blur",
        "crop_out_of_zero_padded",
        "draw_marker_on_np_u8",
        "feathered_paste_for_images_of_the_same_size",
        "get_a_random_xy_where_mask_is_foreground",
        "get_actual_annotations",
        "get_cutout_augmentation",
        "get_cutout_descriptors",
        "get_cutout_from_bounding_box",
        "get_cutouts",
        "get_human_cutout_kinds",
        "get_xy_where_mask_is_background",
        "human_cutout_kinds",
        "is_valid_clip_id",
        "make_fake_segmentation_annotations",
        "maybe_crop_to_nonzero_alpha_region",
        "old_get_cutout_descriptors",
        "paste_cutout_onto_segmentation_annotation",
        "paste_multiple_cutouts_onto_one_camera_posed_segmentation_annotation",
        "scale_image_and_point_together",
        "translated_feathered_paste",
        "translated_paste_onto_blank_canvas",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "mutate_all_jsons = mutate_all_jsons_cli_tool:mutate_all_jsons_cli_tool",
        ],
    },
)
