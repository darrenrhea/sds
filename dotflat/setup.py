from setuptools import setup

setup(
    name="dotflat",
    version="0.1.0",
    py_modules=[
        "full_relative_path_ify_this_mask_file_name_fragment",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "approve_frame_range = approve_frame_range_cli_tool:approve_frame_range_cli_tool",
            "dotflat = dotflat_cli_tool:dotflat_cli_tool",
        ],
    },
)
