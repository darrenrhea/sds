from setuptools import setup

setup(
    name="annotation_utils",
    version="0.1.0",
    py_modules=[
        "get_click_on_image",
        "get_click_on_image_by_two_stage_zoom",
        "get_click_on_resized_image",
        "get_click_on_subrectangle_of_image",
        "get_clicks_on_image",
        "get_clicks_on_image_with_color_confirmation",
        "get_width_and_height_that_fits_on_the_screen",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "annotate_keypoints = annotate_keypoints_cli_tool:annotate_keypoints_cli_tool",
        ],
    },
)
