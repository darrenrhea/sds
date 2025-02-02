from setuptools import setup

setup(
    name="nuke_texture_rendering",
    version="0.1.0",
    py_modules=[
        "decide_color_correction_for_fake_floor_not_floor_annotations",
        "get_t_hit",
        "insert_ad_into_camera_posed_original_video_frame",
        "insert_ads_faster",
        "insert_quads_into_camera_posed_image_behind_mask",
        "least_positive_element_and_index",
        "least_positive_element_and_index_over_last_axis",
        "make_map_from_annotation_id_to_camera_pose",
        "make_map_from_clip_id_and_frame_index_to_camera_pose",
        "make_map_from_clip_id_and_frame_index_to_video_frame_annotation",
        "nuke_texture_rendering",
        "nuke_texture_rendering_demo",
        "nuke_texture_rendering_one_off",
        "perspective_insert_mask",
        "render_ad_on_subregion",
        "render_ad_texture_on_quad",
        "render_ads_on_subregion",
        "render_masks_on_subregion",
        "rip_mask_values_at_these_world_points",
        "rip_world_points",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "mffnfabianfu_make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath = mffnfabianfu_make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath_cli_tool:mffnfabianfu_make_fake_floor_not_floor_annotations_by_inserting_a_new_floor_underneath_cli_tool",
        ],
    },
)
