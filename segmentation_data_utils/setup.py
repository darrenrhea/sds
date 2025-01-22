from setuptools import setup

setup(
    name="segmentation_data_utils",
    version="0.1.0",
    py_modules=[
        "maybe_find_sister_original_path_of_this_mask_path",
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
