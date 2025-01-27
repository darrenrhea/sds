from setuptools import setup

setup(
    name="video_frame_data_utils",
    version="0.1.0",
    py_modules=[
        "get_original_frame_from_clip_id_and_frame_index",
        "get_video_frame_path_from_clip_id_and_frame_index",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "priframe = priframe_cli_tool:priframe_cli_tool",
        ],
    },
)
