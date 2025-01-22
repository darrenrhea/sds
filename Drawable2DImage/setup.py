from setuptools import setup

setup(
    name="Drawable2DImage",
    version="0.0.1",
    py_modules=["Drawable2DImage", "demo_Drawable2DImage",],
    license="MIT",
    long_description=(
        "Draws lines, crosses,and pluses with text annotations onto enlargened photographs"
    ),
    install_requires=[
        "numpy",
        "pillow",
        "better_json",
    ],
    entry_points={
        "console_scripts": ["demo_Drawable2DImage = demo_Drawable2DImage:main"]
    },
)
