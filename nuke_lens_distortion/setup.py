from setuptools import setup

setup(
    name="nuke_lens_distortion",
    version="0.2.0",
    py_modules=["nuke_lens_distortion",],
    license="MIT",
    long_description=("Defines the Nuke lens distortion and its inverse."),
    install_requires=["numpy", "rodrigues_utils",],
)
