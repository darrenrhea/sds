from setuptools import setup

setup(
    name="scorebug_utilities",
    version="0.1.0",
    py_modules=[
        "gather_video_scorebug_assets",
        "scorebug_segmenter",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "gather_scorebugs = gather_scorebugs_cli_tool:gather_scorebugs_cli_tool",
            "jpsosa_just_paste_scorebugs_onto_segmentation_annotations = jpsosa_just_paste_scorebugs_onto_segmentation_annotations_cli_tool:jpsosa_just_paste_scorebugs_onto_segmentation_annotations_cli_tool",
        ],
    },
)
