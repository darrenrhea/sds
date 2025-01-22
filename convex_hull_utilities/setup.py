from setuptools import setup

setup(
    name="convex_hull_utilities",
    version="0.1.0",
    py_modules=[
        "find_2d_convex_hull",
        "find_2d_convex_hull_demo",
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
