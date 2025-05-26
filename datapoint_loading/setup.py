from setuptools import setup

setup(
    name="datapoint_loading",
    version="0.1.0",
    py_modules=[
        "load_datapoints_in_parallel",
        "new_load_datapoints_in_parallel",
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
