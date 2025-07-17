from setuptools import setup

setup(
    name="augmentation_tools",
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
            "show_augmentations = show_augmentations_cli_tool:show_augmentations_cli_tool",
        ],
    },
)
