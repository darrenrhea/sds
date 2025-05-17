from setuptools import setup

setup(
    name="infer_segmentation",
    version="0.1.0",
    py_modules=[
        "FinalModel",
        "check_todo_task",
        "get_final_model_from_id",
        "get_list_of_input_and_output_file_paths_from_json_file_path",
        "get_list_of_input_and_output_file_paths_from_jsonable",
        "get_list_of_input_and_output_file_paths_from_old_style",
        "ifrij_infer_frame_ranges_in_json5",
        "infer_all_the_patches",
        "infer_arbitrary_frames",
        "infer_arbitrary_frames_from_a_clip",
        "infer_from_id",
        "make_frame_ranges_file",
        "parallel_infer3",
        "parse_cli_args_for_inferers",
        "segment_thread",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "infer = infer_cli_tool:infer_cli_tool",
            "isf_infer_specific_frames = isf_infer_specific_frames_cli_tool:isf_infer_specific_frames_cli_tool",
        ],
    },
)
