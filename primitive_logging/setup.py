from setuptools import setup

setup(
    name="primitive_logging",
    version="0.1.0",
    py_modules=[
        "dedent_lines",
        "print_dd",
        "print_dedent",
        "print_error",
        "print_green",
        "print_red",
        "print_warning",
        "print_yellow",
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
