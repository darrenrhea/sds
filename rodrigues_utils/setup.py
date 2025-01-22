from setuptools import setup

setup(
    name="rodrigues_utils",
    version="0.1.0",
    py_modules=["rodrigues_utils",],
    license="MIT",
    long_description=(
        "Utilities to rotate by a Rodrigues angle-axis vector,"
        "and inter-conversions between SO_3 rotation matrices"
        "and the corresponding Rodrigues angle-axis vector."
        "also twist, tilt, pan angles"
    ),
    install_requires=['scipy']
)
