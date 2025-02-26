from setuptools import setup

setup(
    name="cutout_utilities",
    version="0.1.0",
    py_modules=[
        "annotate_scale_on_balls",
        "attempt_to_annotate_a_cutout",
        "get_clip_id_and_frame_index_from_file_name",
        "get_valid_cutout_kinds",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "asoc_annotate_scale_on_cutouts = asoc_annotate_scale_on_cutouts_cli_tool:asoc_annotate_scale_on_cutouts_cli_tool",
            "check_enumeration = check_enumeration_cli_tool:check_enumeration_cli_tool",
            "daac_databaseify_all_approved_cutouts = daac_databaseify_all_approved_cutouts_cli_tool:daac_databaseify_all_approved_cutouts_cli_tool",
            "fix_cutouts_color = fix_cutouts_color_cli_tool:fix_cutouts_color_cli_tool",
            "laacitd_look_at_all_cutouts_in_this_directory = laacitd_look_at_all_cutouts_in_this_directory_cli_tool:laacitd_look_at_all_cutouts_in_this_directory_cli_tool",
        ],
    },
)
