from setuptools import setup

setup(
    name="onnx_stuff",
    version="0.1.0",
    py_modules=[
        "dump_to_onnx",
        "get_cuda_devices",
        "get_list_of_input_and_output_file_paths_from_json_file_path",
        "get_list_of_input_and_output_file_paths_from_jsonable",
        "get_list_of_input_and_output_file_paths_from_old_style",
        "onnx_dumper",
        "onnx_infer",
        "parse_cli_args_for_inferers",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
        ],
    },
)
