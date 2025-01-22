from setuptools import setup

setup(
    name="rsync_utils",
    version="0.1.0",
    py_modules=[
        "download_via_rsync",
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
