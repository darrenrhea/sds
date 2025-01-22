from setuptools import setup

setup(
    name="keypoint_matching_utilities",
    version="0.1.0",
    py_modules=[
        "keypoint_matching_utilities",
        "show_keypoint_matches",
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
