from setuptools import setup

setup(
    name="camera_pose_data",
    version="0.1.0",
    py_modules=[
        "get_camera_pose_from_clip_id_and_frame_index",
        "get_screen_corners_from_clip_id_and_frame_index",
        "validate_one_camera_pose",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "validate_camera_poses = validate_camera_poses_cli_tool:validate_camera_poses_cli_tool",
        ],
    },
)
