from setuptools import setup

setup(
    name="infer_segmentation",
    version="0.1.0",
    py_modules=[
    ],
    license="MIT",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=[
    ],
    entry_points={
        "console_scripts": [
            "infer = infer_cli_tool:infer_cli_tool",
        ],
    },
)
