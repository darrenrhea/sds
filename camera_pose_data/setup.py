from setuptools import setup

setup(
    name="camera_pose_data",
    version="0.1.0",
    py_modules=[
        "arcpttsd_add_raguls_camera_poses_to_the_segmentation_data",
        "asccfn_a_simple_camera_classifier_for_nba",
        "camera_breakdown",
        "camera_type_classification",
        "compute_floor_not_floor_for_frames_with_camera_pose",
        "gacpvfa_get_all_camera_posed_video_frame_annotations",
        "get_camera_pose_from_clip_id_and_frame_index",
        "get_camera_pose_from_json_file_path",
        "get_camera_pose_from_sha256",
        "get_python_internalized_video_frame_annotations",
        "get_screen_corners_from_clip_id_and_frame_index",
        "get_valid_camera_names",
        "make_python_internalized_video_frame_annotation",
        "make_screen_corners_track",
        "validate_one_camera_pose",
        "vovfa_validate_one_video_frame_annotation",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "validate_camera_poses = validate_camera_poses_cli_tool:validate_camera_poses_cli_tool",
            "vavfaftnba_validate_all_video_frame_annotations_for_the_nba = vavfaftnba_validate_all_video_frame_annotations_for_the_nba_cli_tool:vavfaftnba_validate_all_video_frame_annotations_for_the_nba_cli_tool",
        ],
    },
)
