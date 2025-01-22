from setuptools import setup

setup(
    name="better_json",
    version="0.1.0",
    py_modules=[
        "better_json",
        "color_print_json",
        "dump_as_jsonlines",
        "load_json_file",
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
