from setuptools import setup

setup(
    name="custom_pytorch_datasets",
    version="0.1.0",
    py_modules=[
        "WarpDataset",
        "WarpDataset_u16",
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
