from setuptools import setup

setup(
    name="gpu_utilities",
    version="0.1.0",
    py_modules=[
        "get_cuda_devices",
        "get_torch_backend",
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
