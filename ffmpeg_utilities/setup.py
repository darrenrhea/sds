from setuptools import setup

setup(
    name="ffmpeg_utilities",
    version="0.1.0",
    py_modules=[
        "extract_a_segment_of_frames_from_video",
        "extract_single_frame_from_video",
        "extract_from_this_video_these_frames",
        "extract_clip_from_video",
        "extract_all_frames_from_video",
        "extract_every_nth_frame_from_video",
        "concatenate_videos",
        "make_evaluation_video",
        "make_plain_video",
    ],
    license="MIT",
    long_description=(
        "This is a wrapper for using ffmpeg from Python.  extract specific frames from videos via ffmpeg"
    ),
    install_requires=["numpy"],
    entry_points={
        "console_scripts": [
            "afntv_add_frame_numbers_to_video = afntv_add_frame_numbers_to_video_cli_tool:afntv_add_frame_numbers_to_video_cli_tool",
            "mev_make_evaluation_video=mev_make_evaluation_video_cli_tool:mev_make_evaluation_video_cli_tool",
            "mpv_make_plain_video=make_plain_video_cli_tool:make_plain_video_cli_tool",
            "extract_every_nth_frame_from_video=extract_every_nth_frame_from_video:main",
            "esffv_extract_single_frame_from_video=esffv_extract_single_frame_from_video_cli_tool:esffv_extract_single_frame_from_video_cli_tool",
            "eaffv_extract_all_frames_from_video=eaffv_extract_all_frames_from_video_cli_tool:extract_all_frames_from_video_cli_tool",
            "eenffv_extract_every_nth_frame_from_video=eenffv_extract_every_nth_frame_from_video_cli_tool:eenffv_extract_every_nth_frame_from_video_cli_tool",
        ]
    }
)
