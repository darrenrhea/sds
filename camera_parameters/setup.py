from setuptools import setup

setup(
    name="camera_parameters",
    version="0.0.1",
    py_modules=["camera_parameters", "camera_parameters_demo",],
    license="MIT",
    long_description=("Defines a class for storing camera parameters."),
    install_requires=[
        "numpy",
        "rodrigues_utils",
    ],
    entry_points={
        "console_scripts": ["demo_camera_parameters = camera_parameters_demo:main"]
    },
)
