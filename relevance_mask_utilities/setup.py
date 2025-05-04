from setuptools import setup

setup(
    name="relevance_mask_utilities",
    version="0.1.0",
    py_modules=[
        "mrmfsc_make_relevance_masks_from_sentinel_color",
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
