from setuptools import setup

setup(
    name="annotation_utils",
    version="0.1.0",
    py_modules=[
        "get_clicks_on_image",
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "better_json",
        "colorama",
    ],
    entry_points={
        "console_scripts": [
        ],
    },
)
